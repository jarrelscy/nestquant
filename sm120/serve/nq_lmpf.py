"""NQ_LMPF (windowed layer-major 4-bit prefill), worker side (every TP rank). nq_vllm.Runtime.start calls install(rt)
when NQ_LMPF=1; the engine side (nq_lmpf_engine.py) decides the mode of each step and hands it over on the scheduler
output (so.nq_lmpf = dict(m, rid, start, n, last, new, end, budget_s, mnbt) or None), so every rank runs the same order.

Setup (after Worker.compile_or_warm_up_model: KV cache allocated, graphs captured): size the ring and the window buffers
from torch.cuda.mem_get_info() minus NQ_LMPF_RESERVE_GB, patch this rank's V2 GPUModelRunner.execute_model (instance
attribute = outermost wrapper), write /dev/shm/nq_lmpf_ready_<enginepid>_<starttime>_r<rank>.json (W = window tokens
this rank can hold, C = ring records per slot). No ready file = the engine grants and annotates nothing.

Per step (execute_model):
  so.nq_lmpf None              prod path (orig); a held executor pause is released first
  K = ceil(n / mnbt) == 1      'op' mode (full | bud): orig forward with the MoE hook active (no executor pause): per MoE
                               layer the hook takes the live table after the mailbox apply, overrides the rows of the
                               experts loaded into the ring and runs the experts against that table
  K > 1, m full | bud          'exec': window of n tokens = K sub-chunks of <= mnbt (the input buffers / DSA topk buffer
                               are mnbt-sized). Executor paused (no new level ops; landed rows applied; table snapshots),
                               attention metadata re-prepared before every layer call (META=rebuild, default; META=snap
                               = prepare once + deep-copy: WRONG output on GPU, shared planner objects, do not use), TP vote (any rank's
                               setup failure -> every rank runs 'plain'), embeddings into hbuf, then the decoder layers
                               eagerly in layer-outer / sub-chunk-inner order (ORDER=chunk: sub-chunk-outer, debug
                               reference), h / residual in hbuf / rbuf, the shared DSA topk buffer stashed per sub-chunk
                               after indexer layers and restored before shared layers. Final norm per sub-chunk;
                               sub-chunks 0..K-2 are sampled inline (prompt logprobs accumulate, MTP proposes = writes its
                               KV) with the KV connector swapped for the no-op one; the last becomes execute_model_state,
                               so the engine's sample_tokens + kv_connector.post_forward (LMCache save) finish the window.
  m plain (K > 1)              same sub-chunks in chunk order through the normal (compiled) model forward, no hook
Ring: 2 slots x C records of this rank's record file, own nqstream engine (dual drive like the executor). Visit v (one
layer, all its sub-chunks) uses slot v % 2. full: at visit v the reads of visit v+1 are issued into the other slot once
visit v-1's experts finished on the GPU (event / sync), then visit v waits for its own reads; every non-level-4 expert
is loaded (C < needed: static popularity order). bud: at the first call of visit v: router counts -> host sync -> the
top-x non-level-4 routed experts, x = rate * allowance / record bytes, allowance = budget left / visits left (per
request, per rank), loaded synchronously. Failed reads keep the snapshot row (level 2).
Disabled (no patch, no ready file): non-V2 runner, PP, LoRA, aux hidden states, sequence-parallel MoE, llama4 scaling,
encoder-decoder. Stats: /dev/shm/nq_lmpf_stats_<key>_r<rank>.json after every LMPF step."""
import os,sys,time,json,types,dataclasses,copy,logging,atexit,importlib
import numpy as np,torch
import nq_lmpf_engine as EN
log=logging.getLogger('vllm.nestquant.lmpf')
NE=256;TOPK=8;ROW_W=20
RING_RECS=int(os.environ.get('NQ_LMPF_RING_RECS','256'))
META=os.environ.get('NQ_LMPF_META','rebuild')
ORDER=os.environ.get('NQ_LMPF_ORDER','layer')
DBG='/dev/shm/nq_lmpf_dbg'      # in-boot override for A/B and profiling, e.g. "meta=rebuild order=chunk prof=1 tab=base"
def dbg_knobs():
    """(meta, order, extra): extra prof=1 -> CUDA-event timing of the exec layer loop (stats last.prof);
    tab=base -> exec MoE uses the snapshot (prod mixed-level) table instead of the ring table (cost isolation only)"""
    m,o,x=META,ORDER,{}
    try:
        for kv in open(DBG).read().split():
            k,_,v=kv.partition('=')
            if k=='meta' and v in ('snap','rebuild','hyb','hyb2'):m=v
            elif k=='order' and v in ('layer','chunk'):o=v
            elif k in ('prof','tab','syncfree','cop','early','mods'):x[k]=v
    except OSError:pass
    return m,o,x
SYNCFREE=os.environ.get('NQ_LMPF_SYNCFREE','1')=='1'
_SF={}
def syncfree_indexer():
    """vllm indexer.build_prefill_chunk_metadata with the two DCP device->host .item() syncs replaced by the same
    integers from compressed_seq_lens_cpu (get_dcp_local_seq_lens on the host). Returns (module, orig, new) or None.
    Swapped in only while an LMPF exec step prepares metadata (single prefill request, no decodes)."""
    if 'r' in _SF:return _SF['r']
    r=None
    try:
        import inspect,textwrap
        M=importlib.import_module('vllm.v1.attention.backends.mla.indexer');f=M.build_prefill_chunk_metadata
        src=textwrap.dedent(inspect.getsource(f))
        a=("        local_total_seq_lens = int(local_cu_seq_lens[-1].item())\n"
           "        max_local_total_seq_lens = int(local_seq_lens.sum(dim=0).max().item())\n")
        b=("        _lc = get_dcp_local_seq_lens(compressed_seq_lens_cpu[start_idx:end_idx], dcp_world_size, None, cp_kv_cache_interleave_size)\n"
           "        local_total_seq_lens = int(_lc[:, dcp_rank].sum())\n"
           "        max_local_total_seq_lens = int(_lc.sum(dim=0).max())\n"
           "        if _SF_CHECK[0] > 0:\n"
           "            _SF_CHECK[0] -= 1\n"
           "            _t = (int(local_cu_seq_lens[-1].item()), int(local_seq_lens.sum(dim=0).max().item()))\n"
           "            if _t != (local_total_seq_lens, max_local_total_seq_lens):\n"
           "                _SF_LOG.error('NestQuant LMPF: sync-free indexer mismatch %s vs %s, using synced values', _t, (local_total_seq_lens, max_local_total_seq_lens))\n"
           "                _SF_CHECK[0] = 10**9\n"
           "                local_total_seq_lens, max_local_total_seq_lens = _t\n")
        if src.count(a)==1 and src.startswith('def ') and 'get_dcp_local_seq_lens' in M.__dict__:
            g=dict(M.__dict__);g['_SF_CHECK']=[8];g['_SF_LOG']=log;exec(compile(src.replace(a,b),M.__file__,'exec'),g)
            r=(M,f,g['build_prefill_chunk_metadata'])
        else:log.warning('NestQuant LMPF: indexer source mismatch, sync-free chunk metadata off')
    except Exception:log.exception('NestQuant LMPF: sync-free indexer patch failed, off')
    _SF['r']=r;return r
class _swap:
    def __init__(s,on):s.r=syncfree_indexer() if (on and SYNCFREE) else None
    def __enter__(s):
        if s.r:s.r[0].build_prefill_chunk_metadata=s.r[2]
    def __exit__(s,*a):
        if s.r:s.r[0].build_prefill_chunk_metadata=s.r[1]
COP=os.environ.get('NQ_LMPF_COP','0')
EARLY=os.environ.get('NQ_LMPF_EARLY','0')=='1'   # exec full: refill in visit_end. MEASURED WORSE (64K 48 s vs 36 s: the eager host is barely ahead of the GPU, so issuing later starves reads); debug only
COP_OK={'RMSNorm','GemmaRMSNorm','LayerNorm','QuantFP8','SiluAndMul','MulAndSilu','RotaryEmbedding','DeepseekScalingRotaryEmbedding',
        'YaRNScalingRotaryEmbedding','MRotaryEmbedding','FusedAddRMSNorm'}
_COP={};COP_SEEN={}
class _cop:
    """eager exec loop only: vLLM CustomOps (custom_ops=none under inductor -> forward_native = unfused torch ops in eager)
    dispatch to their fused forward_cuda kernels instead; allowlisted classes only.  COP_SEEN counts calls per class."""
    def __init__(s,on):s.on=on;s.orig=None
    def __enter__(s):
        if not s.on:return
        from vllm.model_executor.custom_op import CustomOp
        orig=CustomOp.forward;s.orig=(CustomOp,orig)
        def fwd(self,*a,**k):
            t=type(self);f=_COP.get(t)
            if f is None:
                fc=getattr(t,'forward_cuda',None)
                f=fc if (t.__name__ in COP_OK and fc is not None and fc is not getattr(CustomOp,'forward_cuda',None)) else False
                _COP[t]=f
            COP_SEEN[t.__name__]=COP_SEEN.get(t.__name__,0)+1
            return f(self,*a,**k) if f else orig(self,*a,**k)
        CustomOp.forward=fwd
    def __exit__(s,*a):
        if s.orig:s.orig[0].forward=s.orig[1]
RESERVE_GB=float(os.environ.get('NQ_LMPF_RESERVE_GB','1.5'))
RATE0=float(os.environ.get('NQ_LMPF_RATE0_GBPS','2.0'))*1e9
SNAP_MB=float(os.environ.get('NQ_LMPF_SNAP_MAX_MB','512'))
PAUSE_S=float(os.environ.get('NQ_LMPF_PAUSE_S','30'))
READ_TIMEOUT=float(os.environ.get('NQ_LMPF_READ_TIMEOUT_S','60'))
MIN_C=int(os.environ.get('NQ_LMPF_MIN_RECS','32'))

# ---------------- pure helpers (CPU-tested) ----------------
def split(n,mnbt):
    """[(offset, tokens)] of K = ceil(n / mnbt) sub-chunks: full mnbt chunks then the remainder, i.e. the same
    boundaries as prod chunked prefill.  Full 4096 chunks are also what VLLM_GLM_RAW_KV_GATHER requires (exact
    q.shape 4096); even splits (e.g. 4000) fall back to the per-layer NCCL AG+RS path, ~+25% prefill GPU time."""
    m=max(1,mnbt);q=m;out=[];o=0
    while o<n:m=min(q,n-o);out.append((o,m));o+=m
    return out
def sequence(layers,ring,K,order):
    """layer calls [(L, k, v, first, last)]: v = visit index (one ring use) for ring layers, else None"""
    ri={L:i for i,L in enumerate(ring)};nR=len(ring);out=[]
    if order=='chunk':
        for k in range(K):
            for L in layers:out.append((L,k,k*nR+ri[L] if L in ri else None,True,True))
    else:
        for L in layers:
            for k in range(K):out.append((L,k,ri.get(L),k==0,k==K-1))
    return out
def full_select(lv,C,pop):
    """experts to load for a visit: every non-level-4 expert; more than C -> the C most popular (static n_routed)"""
    need=np.nonzero(np.asarray(lv)!=4)[0]
    if len(need)>C:need=np.sort(need[np.argsort(-np.asarray(pop)[need],kind='stable')[:C]])
    return [int(e) for e in need]
def bud_select(cnt,lv,x,C):
    """top-x routed (count > 0) non-level-4 experts by count"""
    x=min(int(x),C)
    if x<=0:return []
    c=np.where(np.asarray(lv)!=4,np.asarray(cnt),0);cand=np.nonzero(c>0)[0]
    return [int(e) for e in cand[np.argsort(-c[cand],kind='stable')][:x]]
def visits_left(nvis,v,end,start,n):
    """ring visits left for a request: this step's (from v on) + nvis per future step of n tokens"""
    rem=max(0,end-(start+n));return (nvis-v)+(-(-rem//max(n,1)))*nvis
def bud_x(left_s,nleft,rate,rb):
    """(records to load, allowance s) for one visit"""
    a=max(0.,left_s)/max(nleft,1);return int(rate*a//rb),a
def size_plan(avail,rb,per_tok,window,mnbt,recs,min_c=MIN_C):
    """(W, C) that fit in avail bytes: shrink the window first (halving, down to 2 mnbt), then the ring (down to min_c);
    W = 0 (no windows, single-step bud / full only) if even the smallest window does not fit; None if no ring fits"""
    C=min(recs,NE);W=window
    def need(W,C):return 2*C*rb+W*per_tok
    while need(W,C)>avail and W>2*mnbt:W=max(2*mnbt,W//2)
    while need(W,C)>avail and C>min_c:C=max(min_c,C//2)
    if need(W,C)<=avail and W>mnbt:return W,C
    C=min(recs,NE)
    while 2*C*rb>avail and C>min_c:C=max(min_c,C//2)
    return (0,C) if 2*C*rb<=avail else None

def snap(o,limit=None,memo=None,shared=None):
    """clone the tensors / arrays of a metadata tree (dataclasses, dicts, lists, tuples, namedtuples, namespaces);
    tensors above limit bytes and any other object are shared (shared gets their shapes / types)"""
    limit=SNAP_MB*2**20 if limit is None else limit
    memo={} if memo is None else memo;shared=[] if shared is None else shared
    i=id(o)
    if i in memo:return memo[i]
    if isinstance(o,torch.Tensor):
        if o.numel()*o.element_size()<=limit:r=o.clone()
        else:r=o;shared.append(tuple(o.shape))
        memo[i]=r;return r
    if isinstance(o,np.ndarray):r=memo[i]=o.copy();return r
    if o is None or isinstance(o,(int,float,bool,str,bytes,torch.dtype,torch.device)):return o
    if isinstance(o,dict):
        r=type(o)() if type(o) is dict else dict();memo[i]=r
        for k,v in o.items():r[k]=snap(v,limit,memo,shared)
        if type(o) is not dict:
            try:r=memo[i]=type(o)(r)
            except Exception:pass
        return r
    if isinstance(o,list):
        r=[];memo[i]=r;r.extend(snap(v,limit,memo,shared) for v in o);return r
    if isinstance(o,tuple):
        v=[snap(x,limit,memo,shared) for x in o]
        r=type(o)(*v) if hasattr(o,'_fields') else tuple(v);memo[i]=r;return r
    if (dataclasses.is_dataclass(o) and not isinstance(o,type)) or isinstance(o,types.SimpleNamespace):
        r=copy.copy(o);memo[i]=r
        names={f.name for f in dataclasses.fields(o)} if dataclasses.is_dataclass(o) else set()
        names|=set(getattr(o,'__dict__',{}))
        for k in names:
            try:v=getattr(o,k)
            except AttributeError:continue
            nv=snap(v,limit,memo,shared)
            if nv is not v:object.__setattr__(r,k,nv)
        return r
    if any(isinstance(v,torch.Tensor) for v in getattr(o,'__dict__',{}).values()):shared.append(type(o).__name__)
    return o

def write_json(path,d):
    tmp=path+'.tmp'
    with open(tmp,'w') as f:json.dump(d,f)
    os.replace(tmp,path)

# ---------------- ring ----------------
class Ring:
    """2 slots x C records; eng = nqstream engine (or a mock: upgrade / poll); rows[(L, E)] = (r2, z, m) numpy rows
    (executor rc: level-4 row = z + m * record address); rec(L, E) -> record index"""
    def __init__(s,eng,rb,C,dev,rows,rec):
        s.eng=eng;s.rb=rb;s.C=C;s.dev=dev;s.rows=rows;s.rec=rec;cuda=dev.type=='cuda'
        s.buf=torch.empty(2,C,rb,dtype=torch.uint8,device=dev)
        s.sink=torch.zeros(2*C,ROW_W,dtype=torch.int64,device=dev);s.sinkq=torch.zeros(2*C,dtype=torch.int32,device=dev)
        pin=(lambda t:t.pin_memory()) if cuda else (lambda t:t)
        s.rp=[pin(torch.zeros(C,ROW_W,dtype=torch.int64)) for _ in range(2)];s.ip=[pin(torch.zeros(C,dtype=torch.int64)) for _ in range(2)]
        s.rd=torch.zeros(2,C,ROW_W,dtype=torch.int64,device=dev);s.idd=torch.zeros(2,C,dtype=torch.int64,device=dev)
        s.wt=torch.zeros(2,NE,ROW_W,dtype=torch.int64,device=dev)
        s.ev=[torch.cuda.Event(),torch.cuda.Event()] if cuda else None
        s.tag=0;s.q=0;s.pend=[{},{}];s.res=[[],[]];s.nfail=[0,0];s.t0=[0.,0.];s.L=[None,None]
        s.st=dict(recs=0,bytes=0,failed=0,wait_s=0.,issued=0)
    def addr(s,slot,j):return s.buf.data_ptr()+(slot*s.C+j)*s.rb
    def issue(s,slot,L,Es):
        """queue reads of experts Es of layer L into slot (the slot must be idle: waited, and its GPU users done)"""
        assert not s.pend[slot],'ring slot busy'
        assert len(Es)<=s.C,(len(Es),s.C)
        s.res[slot]=[];s.nfail[slot]=0;s.t0[slot]=time.perf_counter();s.L[slot]=L
        sp=s.sink.data_ptr();qp=s.sinkq.data_ptr()
        for j,E in enumerate(Es):
            _,z,m=s.rows[L,E];a=s.addr(slot,j);row=torch.from_numpy(z+m*a);s.tag+=1;s.q=s.q%0x7fffffff+1;i=slot*s.C+j
            s.eng.upgrade(s.tag,s.rec(L,E),a,sp+i*ROW_W*8,row,qp+4*i,s.q);s.pend[slot][s.tag]=(E,row)
        s.st['issued']+=len(Es)
    def poll(s):
        for tag,hit,trd,te in s.eng.poll():
            for sl in (0,1):
                p=s.pend[sl].pop(tag,None)
                if p is None:continue
                if trd<0:s.nfail[sl]+=1;s.st['failed']+=1
                else:s.res[sl].append(p);s.st['recs']+=1;s.st['bytes']+=s.rb
                break
    def wait(s,slot,timeout=READ_TIMEOUT):
        """block until every read of slot landed (or failed); -> (landed [(E, row)], failed count, seconds waited)"""
        t=time.perf_counter()
        while True:
            s.poll()
            if not s.pend[slot]:break
            if time.perf_counter()-t>timeout:raise TimeoutError(f'NQ_LMPF ring: {len(s.pend[slot])} reads of slot {slot} not landed in {timeout:.0f} s')
            time.sleep(2e-4)
        dt=time.perf_counter()-t;s.st['wait_s']+=dt
        return s.res[slot],s.nfail[slot],dt
    def drain(s,timeout=READ_TIMEOUT):
        for sl in (0,1):
            if s.pend[sl]:s.wait(sl,timeout)
            s.res[sl]=[]
    def table(s,slot,base,res):
        """wt[slot] = base with the landed ring rows (stream-ordered; the pinned staging of the slot is reused only after
        the slot's previous GPU users finished, which the visit fences guarantee)"""
        wt=s.wt[slot];wt.copy_(base);n=len(res)
        if n:
            rp=s.rp[slot].numpy();ip=s.ip[slot].numpy()
            for j,(E,row) in enumerate(res):rp[j]=row.numpy();ip[j]=E
            nb=s.dev.type=='cuda'
            s.rd[slot,:n].copy_(s.rp[slot][:n],non_blocking=nb);s.idd[slot,:n].copy_(s.ip[slot][:n],non_blocking=nb)
            wt.index_copy_(0,s.idd[slot,:n],s.rd[slot,:n])
        return wt

# ---------------- worker runtime ----------------
class Prep:
    __slots__=('ib','meta','slots','extra')
    def __init__(s,ib,meta,slots,extra):s.ib=ib;s.meta=meta;s.slots=slots;s.extra=extra
    @property
    def pos(s):return s.extra.get('positions',s.ib.positions)

def _inner(m,depth=0):
    """the decoder stack (DeepseekV2Model-like: layers, norm, embed_input_ids, start_layer / end_layer)"""
    if all(hasattr(m,a) for a in ('layers','norm','embed_input_ids','start_layer','end_layer')):return m
    if depth>4:return None
    for a in ('model','language_model','module','runnable','_model'):
        c=getattr(m,a,None)
        if c is not None and c is not m:
            r=_inner(c,depth+1)
            if r is not None:return r
    return None

class LM:
    def __init__(s,rt):
        s.rt=rt;s.active=False;s.ring=None;s.hbuf=None;s.W=0;s.R=None;s.paused=False;s.prid=None;s.snapt=None;s.lv=None
        s.cur=None;s.op=False;s.mode=None;s.vnext=0;s.wt_cur=None;s.bud={};s.rate=RATE0;s.key=None;s.dead=False
        s.acc={};s.tot={};s.early=EARLY;s.meta=META;s.order=ORDER;s.dx={};s.pev=None;s.pacc=dict(wall=0.,lay=0.,moe=0.,prep=0.,steps=0);s.n=dict(op=0,exec=0,plain=0,fallback=0,visits=0,hook_err=0,steps_full=0,steps_bud=0);s.last={}
        s.fail_logged=0
    # ---- setup ----
    def setup(s,R):
        rt=s.rt;s.R=R;dev=rt.dev
        if type(R).__module__!='vllm.v1.worker.gpu.model_runner':return s._off(f'runner {type(R).__module__} is not the V2 GPU runner')
        if getattr(R,'use_pp',False) or getattr(R,'pp_handler',None) is not None:return s._off('pipeline parallel')
        if getattr(R,'lora_config',None):return s._off('LoRA')
        if getattr(R,'is_encoder_decoder',False):return s._off('encoder-decoder')
        inner=_inner(R.model)
        if inner is None:return s._off('no decoder stack found')
        if getattr(inner,'aux_hidden_state_layers',()):return s._off('aux hidden states')
        if getattr(getattr(inner,'config',None),'llama_4_scaling',None) is not None:return s._off('llama4 scaling')
        layers=list(range(inner.start_layer,inner.end_layer))
        if any(getattr(inner.layers[L],'use_sequence_parallel_moe',False) for L in layers):return s._off('sequence-parallel MoE')
        s.inner=inner;s.layers=layers;s.ring_layers=[L for L in layers if L in rt.lay]
        s.vidx={L:i for i,L in enumerate(s.ring_layers)}
        s.mnbt=int(getattr(R,'max_num_tokens',0) or R.vllm_config.scheduler_config.max_num_batched_tokens)
        s.tib=getattr(inner,'topk_indices_buffer',None)
        s.has_idx={}
        for L in layers:
            a=getattr(inner.layers[L],'self_attn',None)
            s.has_idx[L]=None if a is None or not hasattr(a,'indexer') else a.indexer is not None
        H=int(getattr(inner,'hidden_size',0) or inner.config.hidden_size);s.H=H
        s.dtype=getattr(R,'dtype',None) or (inner.embed_tokens.weight.dtype if hasattr(inner,'embed_tokens') else torch.bfloat16)
        s.ffn=getattr(sys.modules.get(type(inner).__module__),'fused_ar_rms_norm',None)
        s.fuse_final=bool(getattr(inner,'pcie_fuse_final_norm',False))
        if s.fuse_final and s.ffn is None:return s._off('pcie_fuse_final_norm without fused_ar_rms_norm')
        esz=torch.empty(0,dtype=s.dtype).element_size()
        per_tok=2*H*esz+(s.tib.shape[1]*4 if s.tib is not None else 0)
        rb=rt.rf.rb if hasattr(rt,'rf') else 0
        free=torch.cuda.mem_get_info(dev)[0];avail=free-RESERVE_GB*2**30
        p=size_plan(avail,rb,per_tok,EN.WINDOW,s.mnbt,RING_RECS) if rb and hasattr(rt,'X') else None
        if p is None and rb and hasattr(rt,'X'):log.warning('NestQuant LMPF rank %d: %.2f GiB free (reserve %.1f): no ring fits',rt.rank,free/2**30,RESERVE_GB)
        W,C=p if p is not None else (0,0)
        try:
            if C:
                import nq_vllm
                IO=nq_vllm._io_cfg(rt.rank,os.environ['NQ_REPACK'],rb);kw={k:v for k,v in IO['kw'].items() if k in ('alt_path','qd_alt')}
                qd=IO['qd'];nh=int(os.environ.get('NQ_LMPF_NHOST') or max(32,2*(qd+kw.get('qd_alt',0))))
                eng=rt.rf.engine(nh,qd,dev.index,**kw)
                s.ring=Ring(eng,rb,C,dev,rt.X.rc,rt.rf.rec)
            if W:
                s.hbuf=torch.empty(W,H,dtype=s.dtype,device=dev);s.rbuf=torch.empty(W,H,dtype=s.dtype,device=dev)
                s.stash=torch.empty(W,s.tib.shape[1],dtype=s.tib.dtype,device=dev) if s.tib is not None else None
            s.W=W
        except Exception:
            log.exception('NestQuant LMPF rank %d: ring / window allocation failed',rt.rank)
            s.ring=None;s.hbuf=s.rbuf=s.stash=None;s.W=0;torch.cuda.empty_cache()
        s.pop={L:np.zeros(NE) for L in s.ring_layers}
        try:
            fj=json.load(open(os.environ.get('NQ_HOME','/nq')+'/threads/22-boundary-experts/fixed_set.json'))
            for L in s.ring_layers:s.pop[L]=np.asarray(fj['n_routed'][str(L)],dtype=np.float64)
        except Exception as e:log.warning('NestQuant LMPF: no static popularity (%s), ring overflow picks by index',e)
        s.acc={m:torch.zeros((),dtype=torch.int64,device=dev) for m in ('full','bud')}
        s.tot={'full':0,'bud':0}
        orig=R.execute_model;lm=s
        @torch.inference_mode()
        def execute_model(scheduler_output,*a,**k):return lm.step(orig,scheduler_output,a,k)
        R.execute_model=execute_model
        try:s.key=EN.boot_key(os.getppid())
        except Exception as e:log.warning('NestQuant LMPF: no boot key (%s)',e)
        s.rp=EN.ready_path(s.key,rt.rank) if s.key else None
        s.sp=f'/dev/shm/nq_lmpf_stats_{s.key}_r{rt.rank}.json' if s.key else None
        if s.rp and (s.W or s.ring is not None):
            write_json(s.rp,dict(W=s.W,C=C if s.ring is not None else 0,mnbt=s.mnbt,ring=s.ring is not None,rank=rt.rank,t=time.time()))
            atexit.register(lambda p=s.rp:os.path.exists(p) and os.remove(p))
        log.info('NestQuant LMPF rank %d: on, window %d tokens (mnbt %d), ring %s, %d ring layers, %.2f GiB free before, meta %s, order %s, ready %s',
                 rt.rank,s.W,s.mnbt,f'{C} recs x 2 ({2*C*rb/2**30:.2f} GiB)' if s.ring is not None else 'off',len(s.ring_layers),free/2**30,META,ORDER,s.rp)
    def _off(s,why):
        log.warning('NestQuant LMPF rank %d: off (%s)',s.rt.rank,why);s.dead=True;return None
    # ---- executor pause ----
    def pause(s):
        if s.paused:return
        rt=s.rt
        if hasattr(rt,'X'):
            with rt.cv:
                rt.ncap+=1
                ok=rt.cv.wait_for(lambda:not rt.in_iter and not rt.X.ops,timeout=PAUSE_S)
            if not ok:log.warning('NestQuant LMPF rank %d: executor not idle after %.0f s (%d ops in flight), snapshot anyway',rt.rank,PAUSE_S,len(rt.X.ops))
        s.paused=True
        for L in s.ring_layers:rt.lay[L]['MB'].apply()
        s.snapt={L:rt.lay[L]['M'].table.clone() for L in s.ring_layers}
        s.lv=torch.stack([s.snapt[L][:,0] for L in s.ring_layers]).cpu().numpy() if s.ring_layers else None
    def unpause(s):
        if not s.paused:return
        rt=s.rt;s.paused=False;s.snapt=None;s.lv=None;s.prid=None
        if hasattr(rt,'X'):
            with rt.cv:rt.ncap-=1;rt.cv.notify_all()
    # ---- MoE hook (nq_vllm.forward) ----
    def moe(s,L,d,M,x,xh,w,ids):
        if s.ring is None or torch.cuda.is_current_stream_capturing():return None
        if s.op:
            v=s.vidx.get(L)
            if v is None:return None
            first=last=True
            if v!=s.vnext:s.skip_to(v)
        else:
            c=s.cur
            if c is None or c[0]!=L or c[2] is None:return None
            v,first,last=c[2],c[3],c[4]
        try:
            if first:
                if s.op:d['MB'].apply()
                s.wt_cur=s.vstart(v,L,M,ids)
            wt=s.wt_cur
        except Exception:
            s.n['hook_err']+=1
            if s.fail_logged<5:s.fail_logged+=1;log.exception('NestQuant LMPF rank %d: ring hook failed at L%d, ring off',s.rt.rank,L)
            s.ring=None;return None
        if not s.op and s.dx.get('tab')=='base':wt=s.snapt[L]
        lv4=wt[:,0]==4;s.acc[s.mode].add_(lv4[ids].sum());s.tot[s.mode]+=ids.numel()
        pe=s.pev
        if pe is not None:e0=torch.cuda.Event(enable_timing=True);e0.record()
        out=_moe_tab(M,x,xh,w,ids,wt)
        if pe is not None:e1=torch.cuda.Event(enable_timing=True);e1.record();pe['moe'].append((e0,e1))
        if last:
            if s.ring.ev is not None:s.ring.ev[v%2].record()
            s.vnext=v+1;s.n['visits']+=1
            if s.early and not s.op and s.mode=='full':s.visit_end(v)
        return out
    def skip_to(s,v):
        """op mode: a visit was not seen (should not happen): its prefetched reads are dropped"""
        s.ring.drain();s.vnext=v;s.pref=None
    def vstart(s,v,L,M,ids):
        R=s.ring;sl=v%2;base=s.snapt[L] if not s.op else M.table
        if s.mode=='full':
            nv=v+1<s.nvis;Ln=s.vis[v+1] if nv else None
            if s.op:
                lvn=s.rt.lay[Ln]['M'].table[:,0].cpu().numpy() if nv else None   # host sync = fence of visit v-1 too
                if nv:R.issue((v+1)%2,Ln,full_select(lvn,R.C,s.pop[Ln]))
            elif not s.early and nv:
                if v>0 and R.ev is not None:
                    tq=time.perf_counter();R.ev[(v-1)%2].synchronize();s.pacc['vsync']=s.pacc.get('vsync',0.)+time.perf_counter()-tq
                R.issue((v+1)%2,Ln,full_select(s.lv[s.vidx[Ln]],R.C,s.pop[Ln]))
            # EARLY (exec): visit v+1 was issued at the end of visit v-1's launches (visit_end), a full layer ahead of the GPU
            res,nf,dt=R.wait(sl) if R.L[sl]==L else ([],0,0.)
            s.last['stall_s']=s.last.get('stall_s',0.)+dt
        else:
            cl=torch.stack([torch.bincount(ids.flatten(),minlength=NE)[:NE],(base[:,0]==4).long()]).cpu().numpy()   # sync: fences slot sl
            st=s.bst;left=visits_left(s.nvis,v,st['end'],st['start'],st['n'])
            x,allow=bud_x(s.bud.get(st['rid'],0.),left,s.rate,R.rb)
            Es=bud_select(cl[0],np.where(cl[1]>0,4,2),x,R.C);res=[]
            if Es:
                t=time.perf_counter();R.issue(sl,L,Es);res,nf,dt=R.wait(sl);el=time.perf_counter()-t
                s.bud[st['rid']]=s.bud.get(st['rid'],0.)-el;s.last['bud_s']=s.last.get('bud_s',0.)+el
                if len(res)>=4 and el>0:s.rate=.7*s.rate+.3*max(len(res)*R.rb/el,2e8)
            s.last['bud_recs']=s.last.get('bud_recs',0)+len(res)
        return R.table(sl,base,res)
    def visit_end(s,v):
        """exec full mode, host has launched all of visit v: refill the slot of visit v-1 with visit v+1's reads.  The host waits
        for the GPU to finish visit v-1 here, while the GPU still has all of visit v queued (no bubble); the old place
        (vstart of v, chunk 0) left the GPU with only chunk 0's attention part queued during the sync + issue + table."""
        R=s.ring;w=v+1
        if w>=s.nvis or (w==1 and s.pre1):return
        if v>0 and R.ev is not None:
            tq=time.perf_counter();R.ev[(v-1)%2].synchronize();s.pacc['vsync']=s.pacc.get('vsync',0.)+time.perf_counter()-tq
        Ln=s.vis[w];R.issue(w%2,Ln,full_select(s.lv[s.vidx[Ln]],R.C,s.pop[Ln]))
    def begin(s,info,K,op):
        """ring plan of a step; full: reads of visit 0 issued now"""
        s.mode=info['m'];s.op=op;s.vnext=0;s.wt_cur=None
        s.vis=s.ring_layers if (op or s.order!='chunk') else s.ring_layers*K
        s.nvis=len(s.vis);s.bst=info
        if s.mode=='bud' and info['rid'] not in s.bud:
            s.bud[info['rid']]=float(info.get('budget_s',0.))
            if len(s.bud)>256:
                for k in list(s.bud)[:128]:s.bud.pop(k,None)
        if s.ring is None:return
        if s.ring.ev is not None:torch.cuda.current_stream().synchronize()   # previous users of both slots are done
        s.ring.drain()
        if s.mode=='full' and s.nvis:
            L0=s.vis[0];lv0=s.rt.lay[L0]['M'].table[:,0].cpu().numpy() if op else s.lv[s.vidx[L0]]
            s.ring.issue(0,L0,full_select(lv0,s.ring.C,s.pop[L0]))
            s.pre1=False
            if s.early and not op and s.nvis>1:          # both slots idle: visit 1 too
                L1=s.vis[1];s.ring.issue(1,L1,full_select(s.lv[s.vidx[L1]],s.ring.C,s.pop[L1]));s.pre1=True
    def end(s):
        if s.ring is not None:
            try:s.ring.drain()
            except Exception:log.exception('NestQuant LMPF rank %d: ring drain failed, ring off',s.rt.rank);s.ring=None
        s.active=False;s.cur=None;s.op=False
    # ---- step ----
    def step(s,orig,so,a,k):
        info=None if k.get('dummy_run',False) else getattr(so,'nq_lmpf',None)
        if info is None or (s.prid is not None and info.get('rid')!=s.prid):s.unpause()
        if info is None:return orig(so,*a,**k)
        t0=time.perf_counter();s.last=dict(m=info['m'],n=so.total_num_scheduled_tokens,start=info['start'])
        n=so.total_num_scheduled_tokens;K=len(split(n,s.mnbt));s.last['K']=K
        try:
            if K==1:
                if info['m'] not in ('full','bud') or s.ring is None:return orig(so,*a,**k)
                s.n['op']+=1;s.n['steps_'+info['m']]+=1;s.begin(info,1,True);s.active=True
                try:return orig(so,*a,**k)
                finally:s.end()
            if info['m']=='plain':
                s.n['plain']+=1;return s.run_plain(so,info,prefixed=False)
            s.n['exec']+=1;s.n['steps_'+info['m']]+=1
            s.meta,s.order,s.dx=dbg_knobs()
            s.early=s.dx.get('early','1' if EARLY else '0')=='1'
            with _swap(s.dx.get('syncfree','1')=='1'),_cop(s.dx.get('cop',COP)=='1'):return s.run_exec(so,info)
        finally:
            s.last['t_s']=time.perf_counter()-t0
            if info.get('last'):s.unpause()
            s.publish()
    def prefix(s,so):
        R=s.R
        R.update_pp_decode_requests();R.finish_requests(so);R.free_states(so);R.add_requests(so);R.update_requests(so)
        R.block_tables.apply_staged_writes()
        if getattr(so,'scheduled_new_reqs',None):
            pf=sys.modules.get('nq_pfblock')
            if pf is not None and hasattr(pf,'PI'):pf.PI.new=True
    def prep(s,so,rid,idx,nc,nk,attn=True,state_only=False):
        from vllm.config.compilation import CUDAGraphMode
        from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
        from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
        R=s.R;rs=R.req_states
        rs.num_computed_tokens_np[idx]=nc
        rs.num_computed_prefill_tokens[idx]=min(nc,int(rs.prefill_len.np[idx]))
        rs.num_computed_tokens.gpu[idx:idx+1].fill_(nc)
        if state_only:return None
        fso=types.SimpleNamespace(total_num_scheduled_tokens=nk,num_scheduled_tokens={rid:nk},scheduled_spec_decode_tokens={},
                                  has_structured_output_requests=getattr(so,'has_structured_output_requests',False),
                                  scheduled_encoder_inputs={},finished_req_ids=set(),scheduled_new_reqs=[])
        ib=R.prepare_inputs(fso,BatchExecutionDescriptor(cg_mode=CUDAGraphMode.NONE,num_tokens=nk,num_reqs=1))
        bt,sm=R.prepare_attn(ib)
        R.model_state.preprocess_state(ib,bt,R.kv_cache_config,rs.num_computed_tokens.gpu)
        slots=build_slot_mappings_by_layer(sm,R.kv_cache_config)
        meta=R.model_state.prepare_attn(ib,CUDAGraphMode.NONE,bt,sm,R.attn_groups,R.kv_cache_config) if attn else None
        extra=dict(R.model_state.prepare_inputs(ib,rs) or {})
        return Prep(ib,meta,slots,extra)
    def fctx(s,P,nk):
        from vllm.config.compilation import CUDAGraphMode
        from vllm.forward_context import set_forward_context,BatchDescriptor
        return set_forward_context(P.meta,s.R.vllm_config,num_tokens=nk,num_tokens_across_dp=None,cudagraph_runtime_mode=CUDAGraphMode.NONE,
                                   batch_descriptor=BatchDescriptor(num_tokens=nk),slot_mapping=P.slots,is_padding=P.ib.is_padding)
    def embed(s,P,nk):
        R=s.R
        if getattr(R,'supports_mm_inputs',False) and getattr(R,'is_first_pp_rank',True):
            e=R.model_state.get_mm_embeddings({},P.ib,R.req_states)
            if e is not None:return e[:nk]
        return s.inner.embed_input_ids(P.ib.input_ids[:nk])
    def sample_sub(s,P,hs,final,so):
        """sub-chunk output: inline sample (k < K-1, no-op KV connector) or execute_model_state (last)"""
        from vllm.v1.worker.gpu.model_runner import ExecuteModelState
        from vllm.v1.worker.gpu.kv_connector import NO_OP_KV_CONNECTOR
        R=s.R
        R.execute_model_state=ExecuteModelState(input_batch=P.ib,attn_metadata=P.meta,slot_mappings_by_layer=P.slots,hidden_states=hs,
                                                aux_hidden_states=None,finished_req_ids=so.finished_req_ids if final else set())
        if final:return None
        kc=R.kv_connector;R.kv_connector=NO_OP_KV_CONNECTOR
        try:return R.sample_tokens(None)
        finally:R.kv_connector=kc
    def run_plain(s,so,info,prefixed):
        """sub-chunks in order through the normal model forward (prod MoE path)"""
        R=s.R
        if not prefixed:s.prefix(so)
        rid=info['rid'];idx=R.req_states.req_id_to_index[rid];c0=int(R.req_states.num_computed_tokens_np[idx])
        offs=split(so.total_num_scheduled_tokens,s.mnbt);keep=[]
        for k,(o,nk) in enumerate(offs):
            P=s.prep(so,rid,idx,c0+o,nk)
            with s.fctx(P,nk):
                if k==0:R.kv_connector.pre_forward(so)
                ids=P.ib.input_ids;emb=None
                if getattr(R,'supports_mm_inputs',False):
                    emb=R.model_state.get_mm_embeddings({},P.ib,R.req_states)
                    if emb is not None and not R.model.requires_raw_input_tokens:ids=None
                hs=R.model(**{'input_ids':ids,'positions':P.ib.positions,'inputs_embeds':emb,'intermediate_tensors':None,**P.extra})
            if k<len(offs)-1:keep.append(s.sample_sub(P,hs,False,so))
            else:s.sample_sub(P,hs,True,so)
        return None
    def vote(s,ok):
        try:
            from vllm.distributed import get_tp_group
            g=get_tp_group()
            if g.world_size<=1:return ok
            t=torch.tensor([0 if ok else 1],dtype=torch.int32,device=s.rt.dev);t=g.all_reduce(t)
            return int(t.item())==0
        except Exception:
            log.exception('NestQuant LMPF: TP vote failed');raise
    def run_exec(s,so,info):
        R=s.R;s.prefix(so);s.last.update(meta=s.meta,order=s.order,**s.dx)
        rid=info['rid'];idx=R.req_states.req_id_to_index[rid];c0=int(R.req_states.num_computed_tokens_np[idx])
        n=so.total_num_scheduled_tokens;offs=split(n,s.mnbt);K=len(offs)
        if c0!=info['start'] and s.fail_logged<5:s.fail_logged+=1;log.warning('NestQuant LMPF: worker start %d != scheduler start %d (%s)',c0,info['start'],rid)
        ok=True;snaps=None
        try:
            if s.hbuf is None or n>s.hbuf.shape[0]:raise RuntimeError(f'window {n} > buffers {0 if s.hbuf is None else s.hbuf.shape[0]}')
            if s.ring is not None:
                if s.prid!=rid:s.unpause()
                s.pause();s.prid=rid
            snaps=[]
            if s.meta!='rebuild':
                for o,nk in offs:
                    sh=[];snaps.append(snap(s.prep(so,rid,idx,c0+o,nk),shared=sh))
                    if sh and s.fail_logged<5:s.fail_logged+=1;log.warning('NestQuant LMPF: metadata snapshot shares %s',sh[:8])
        except Exception:
            log.exception('NestQuant LMPF rank %d: window setup failed',s.rt.rank);ok=False
        if not s.vote(ok):
            s.n['fallback']+=1;s.unpause()
            return s.run_plain(so,info,prefixed=True)
        return s.layer_run(so,info,rid,idx,c0,offs,snaps)
    def layer_run(s,so,info,rid,idx,c0,offs,snaps):
        R=s.R;inner=s.inner;K=len(offs);tib=s.tib;stash=s.stash;hb=s.hbuf;rbf=s.rbuf
        def getP(k):
            c,nk=c0+offs[k][0],offs[k][1]
            if s.meta=='rebuild':return s.prep(so,rid,idx,c,nk)
            if s.meta=='hyb':q=s.prep(so,rid,idx,c,nk,attn=False);return Prep(q.ib,snaps[k].meta,q.slots,q.extra)
            if s.meta=='hyb2':s.prep(so,rid,idx,c,nk,state_only=True)
            return snaps[k]
        for k,(o,nk) in enumerate(offs):        # embeddings (vocab-parallel: collective, after the vote)
            P=s.prep(so,rid,idx,c0+o,nk)
            with s.fctx(P,nk):hb[o:o+nk].copy_(s.embed(P,nk))
        if s.ring is not None:s.begin(info,K,False)
        else:s.mode=info['m'];s.op=False
        seq=sequence(s.layers,s.ring_layers,K,s.order);first=s.layers[0];pre=False
        from vllm.forward_context import get_forward_context
        prof=s.dx.get('prof')=='1';s.pev=dict(moe=[],lay=[]) if prof else None
        hks=[];mev={}
        if prof and s.dx.get('mods')=='1':      # per-submodule CUDA-event GPU time (depth <= 3, mlp internals excluded)
            def _pre(nm):
                def f(m,a):e=torch.cuda.Event(enable_timing=True);e.record();mev.setdefault(nm,[]).append([e,None])
                return f
            def _post(nm):
                def f(m,a,o):e=torch.cuda.Event(enable_timing=True);e.record();mev[nm][-1][1]=e
                return f
            for L in s.layers:
                for nm,m in inner.layers[L].named_modules():
                    if not nm or nm.count('.')>2 or nm.startswith('mlp.'):continue
                    hks+=[m.register_forward_pre_hook(_pre(nm)),m.register_forward_hook(_post(nm))]
        if prof:torch.cuda.synchronize();tw=time.perf_counter()
        try:
            s.active=True
            for L,k,v,fst,lst in seq:
                o,nk=offs[k];tp=time.perf_counter() if prof else 0;P=getP(k);layer=inner.layers[L];hi=s.has_idx.get(L)
                if prof:s.pacc['prep']+=time.perf_counter()-tp
                with s.fctx(P,nk):
                    get_forward_context().all_moe_layers=None
                    if not pre:R.kv_connector.pre_forward(so);pre=True
                    if tib is not None and hi is not True and L!=first:tib[:nk].copy_(stash[o:o+nk])
                    s.cur=(L,k,v,fst,lst)
                    if prof:a0=torch.cuda.Event(enable_timing=True);a0.record()
                    h,r=layer(P.pos,hb[o:o+nk],None if L==first else rbf[o:o+nk],None)
                    if prof:a1=torch.cuda.Event(enable_timing=True);a1.record();s.pev['lay'].append((a0,a1))
                    s.cur=None
                    if tib is not None and hi is not False:stash[o:o+nk].copy_(tib[:nk])
                    hb[o:o+nk].copy_(h);rbf[o:o+nk].copy_(r)
        finally:
            s.end()
        if prof:
            torch.cuda.synchronize();wall=time.perf_counter()-tw;pe=s.pev;s.pev=None
            sm=lambda l:sum(a.elapsed_time(b) for a,b in l)/1e3
            pr=s.pacc
            pr['wall']+=wall;pr['lay']+=sm(pe['lay']);pr['moe']+=sm(pe['moe']);pr['steps']+=1
            for h in hks:h.remove()
            for nm,l in mev.items():pr['m:'+nm]=pr.get('m:'+nm,0.)+sm([x for x in l if x[1] is not None])
        for k,(o,nk) in enumerate(offs):
            P=s.prep(so,rid,idx,c0+o,nk)
            with s.fctx(P,nk):
                h,r=hb[o:o+nk],rbf[o:o+nk]
                hs=s.ffn(h,r,inner.norm)[0] if s.fuse_final else inner.norm(h,r)[0]
            s.sample_sub(P,hs,k==K-1,so)
        return None
    # ---- stats ----
    def stats(s):
        sh={m:(int(s.acc[m].item())/s.tot[m] if s.tot.get(m) else None) for m in s.acc}
        return dict(n=dict(s.n),share4=sh,routes=dict(s.tot),rate_gbps=s.rate/1e9,ring=dict(s.ring.st) if s.ring is not None else None,
                    W=s.W,paused=s.paused,last=s.last,prof=s.pacc,t_wall=time.time())
    def publish(s):
        if s.sp is None:return
        try:d=s.stats();d['cop_seen']=dict(COP_SEEN);write_json(s.sp,d)
        except Exception:
            if s.fail_logged<5:s.fail_logged+=1;log.exception('NestQuant LMPF: stats publish failed')

def _moe_tab(M,x,xh,w,ids,tab):
    """nq_vllm.forward's expert paths against an override table (no PFB / lookahead / capture hooks)"""
    import nq_vllm as NV
    T=x.shape[0]
    if T<=NV.BMAX:return M(xh,ids,w,cfg_gu=NV.CFG_GU[T],cfg_dn=NV.CFG_DN[T],table=tab).to(x.dtype)
    if T>=NV.PF_MIN:return M.prefill(xh,ids,w,table=tab).to(x.dtype)
    out=torch.empty(T,x.shape[1],dtype=torch.float32,device=x.device)
    for i in range(0,T,NV.BMAX):
        j=min(T,i+NV.BMAX);M(xh[i:j],ids[i:j],w[i:j],out=out[i:j],cfg_gu=NV.CFG_GU[j-i],cfg_dn=NV.CFG_DN[j-i],table=tab)
    return out.to(x.dtype)

def install(rt):
    """Runtime.start (worker, during weight loading): set up after Worker.compile_or_warm_up_model"""
    if not EN.ON:return None
    lm=LM(rt)
    try:GW=importlib.import_module('vllm.v1.worker.gpu_worker')
    except Exception as e:log.warning('NestQuant LMPF: no gpu_worker (%s), off',e);return None
    Wk=GW.Worker
    if getattr(Wk,'_nq_lmpf',False):return lm
    c0=Wk.compile_or_warm_up_model
    def compile_or_warm_up_model(self,*a,**k):
        r=c0(self,*a,**k)
        try:lm.setup(self.model_runner)
        except Exception:log.exception('NestQuant LMPF rank %d: setup failed, off',rt.rank);lm.dead=True
        return r
    import functools;functools.update_wrapper(compile_or_warm_up_model,c0)
    Wk.compile_or_warm_up_model=compile_or_warm_up_model;Wk._nq_lmpf=True
    return lm
