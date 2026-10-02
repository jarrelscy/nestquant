"""nq-prefill: prefill-borrow, EngineCore side (vLLM scheduler process). Loaded by sitecustomize.py only when
NQ_PREFILL_BORROW=1; patches vllm.v1.core.sched.scheduler.Scheduler.schedule and KVCacheManager.allocate_slots.

While exactly one request runs and it is prefilling with >= NQ_PB_MIN_NEW (8192) prompt tokens still to compute (an
LMCache / prefix hit counts as computed, so a restore never borrows), FREE KV blocks are taken out of the free queue
(ref_cnt 1, prefix hash evicted, in no request's block table: the KV connector and the model never touch them) and
lent to the NQ expert slot pool. Reserve kept free: the rest of the prompt + spec lookahead + NQ_PB_MARGIN blocks, so
the request itself never runs short (1M context: nothing to lend, nothing borrowed). Every SchedulerOutput carries
the full state as attribute nq_pb = (epoch, 'borrow'|'drain', ((b0, b1), ...), n_slots) or None; the workers (nq_pb.py)
carve slots at the first output of an epoch and give them back at the first output without it, in their
execute_model hook before the forward (so before KV writes and before LMCache start_load_kv of that step).
Release: the step after the last prefill chunk (first decode step), the request leaves / another one is scheduled,
or allocate_slots runs short (released inside that schedule() call, then retried: the same output says None).
NQ_PREFILL_SLOTS (155) floating experts per layer wanted during prefill -> extra slots (NQ_PREFILL_SLOTS - nf) x layers;
the block count is sized from the KV tensors' page sizes with the workers' rank-independent carve rule (carve_count).
In-boot off switch: while /dev/shm/nq_pb_off exists no new borrow starts (NQ_PB_OFF).

Phase 2, NQ_PREFILL_KV_OFFLOAD=1 (also needs NQ_PREFILL_BORROW=1): when the free blocks can't give the target (long
context), from the request's 2nd scheduled step on (its LMCache load, if any, is done) and with >= NQ_PB_MIN_NEW prompt
tokens still to go, the USED KV of n_off MLA layers is moved to pinned host memory and streamed back per layer
(nq_pb.KVOff): those layers' whole KV storages, minus two that rotate as staging, become expert slots. The engine
takes no blocks; it evicts the prefix hash of every cached free block (their GPU content is overwritten; LMCache keeps
its copy) and sends kv = dict(n_cand, n_off, rows, blocks, wb) on every output (the request's MLA-group block ids, the
ones this step writes). n_off is sized at each start from /proc/meminfo MemAvailable for ALL TP ranks (host copy =
world x n_off x rows x page; NQ_PB_RAM_FLOOR_GB (38) must stay available) and NQ_PB_KV_HOST_GB per rank (8). Released
like phase 1, plus: a step of <= max cudagraph capture size tokens (piecewise graphs bake KV pointers), the block list
outgrowing rows, MemAvailable < floor mid-borrow (then phase 1 may run for the rest of that request)."""
import os,sys,json,logging,functools
ON=os.environ.get('NQ_PREFILL_BORROW','0')=='1'
MIN_NEW=int(os.environ.get('NQ_PB_MIN_NEW','8192'));PF_NF=int(os.environ.get('NQ_PREFILL_SLOTS','155'))
MARGIN=int(os.environ.get('NQ_PB_MARGIN','16'));OFF=os.environ.get('NQ_PB_OFF','/dev/shm/nq_pb_off')
KV_OFF=os.environ.get('NQ_PREFILL_KV_OFFLOAD','0')=='1'
HOST_GB=min(float(os.environ.get('NQ_PB_KV_HOST_GB','8')),64.);FLOOR_GB=float(os.environ.get('NQ_PB_RAM_FLOOR_GB','38'))
ALIGN=4096
log=logging.getLogger('vllm.nestquant.pb')   # child of vllm's logger (its handler / level); no vllm import at site time

def carve_count(nblk,pages,rb):
    """slots a run of nblk consecutive blocks yields over all KV storages: per storage floor((run bytes - ALIGN) / rb)
    (start aligned up to ALIGN: the same count on every rank whatever the base addresses)"""
    return sum(max(0,(nblk*p-ALIGN)//rb) for p in pages)

def mem_avail():
    try:
        for ln in open('/proc/meminfo'):
            if ln.startswith('MemAvailable:'):return int(ln.split()[1])*1024
    except Exception:pass
    return 0

def layer_idx(name):
    v=[int(x) for x in name.split('.') if x.isdigit()]
    return v[0] if len(v)==1 else None

def mla_candidate(name,n_layers):
    """same rule on the engine (layer names) and the workers: a main-model MLA attention layer (not the indexer cache,
    not the MTP layer); the storage must also be its alone"""
    if 'indexer' in name or 'k_cache' in name or not name.endswith('attn'):return False
    i=layer_idx(name);return i is not None and i<n_layers

def nf0():
    return 77 if os.environ.get('NQ_PREDICTOR','') in ('joint','jf','tf') else 51

def target_extra():
    """(extra slots wanted, rec bytes) from the rank-0 record index"""
    rp=os.environ.get('NQ_REPACK','')
    try:d=json.load(open(f'{rp}/rank0.json'))
    except Exception as e:log.warning('NestQuant prefill-borrow: no %s/rank0.json (%s), off',rp,e);return 0,0
    nL=len(d['layers']);only=os.environ.get('NQ_LAYERS')
    if only:
        a,b=(only.split('-')+[only])[:2];nL=sum(1 for L in d['layers'] if int(a)<=int(L)<=int(b))
    return max(0,(PF_NF-nf0())*nL),int(d['rec_bytes'])

class State:
    def __init__(s,sched):
        s.ep=0;s.phase=None;s.rid=None;s.blocks=[];s.runs=();s.nslots=0;s.mode=0;s.kv=None;s.tried=set();s.tried2=set()
        s.seen=set();s.dead=False;s.n=dict(borrows=0,pressure=0,blocks=0,slots=0,kv_borrows=0)
        s.want,s.rb=target_extra()
        cfg=sched.kv_cache_config;nb=cfg.num_blocks;pg=[];s.nb=nb
        for t in cfg.kv_cache_tensors:
            if getattr(t,'block_stride',0):log.warning('NestQuant prefill-borrow: packed KV tensors, off');s.dead=True;break
            if t.size%nb:log.warning('NestQuant prefill-borrow: KV tensor size %d not a multiple of %d blocks, off',t.size,nb);s.dead=True;break
            pg.append(t.size//nb)
        s.pages=pg;s.kvm=sched.kv_cache_manager;s.pool=s.kvm.block_pool
        if os.environ.get('NQ_HOSTLOOP','py')=='cpp':log.warning('NestQuant prefill-borrow: NQ_HOSTLOOP=cpp not supported, off');s.dead=True
        if not s.want or not s.rb:s.dead=True
        s.minrun=next((n for n in range(1,nb+1) if carve_count(n,pg,s.rb)>0),nb+1) if pg and s.rb else nb+1
        # phase 2 (KV offload) geometry
        s.kv_ok=False;s.n_cand=0;s.kv_page=0;s.kv_gi=None
        vc=getattr(sched,'vllm_config',None)
        try:
            s.world=int(vc.parallel_config.world_size)
            cc=vc.compilation_config;s.max_graph=int(getattr(cc,'max_cudagraph_capture_size',0) or max(cc.cudagraph_capture_sizes or [0]))
            nL=int(vc.model_config.hf_text_config.num_hidden_layers)
        except Exception as e:
            s.world=4;s.max_graph=512;nL=0
            if KV_OFF:log.warning('NestQuant prefill-borrow: phase 2 geometry unknown (%s), phase 2 off',e)
        if KV_OFF and not s.dead and nL:
            cand=[(t.size//nb,t.shared_by[0]) for t in cfg.kv_cache_tensors if len(t.shared_by)==1 and mla_candidate(t.shared_by[0],nL)]
            pgs={p for p,_ in cand};names={n for _,n in cand}
            gis=[i for i,g in enumerate(cfg.kv_cache_groups) if names & set(g.layer_names)]
            if len(cand)>=3 and len(pgs)==1 and len(gis)==1 and names<=set(cfg.kv_cache_groups[gis[0]].layer_names):
                s.kv_ok=True;s.n_cand=len(cand);s.kv_page=pgs.pop();s.kv_gi=gis[0]
            else:log.warning('NestQuant prefill-borrow: phase 2 off: %d MLA candidates, pages %s, groups %s',len(cand),sorted(pgs),gis)
        log.info('NestQuant prefill-borrow: %s; want %d extra slots (%d/layer), rb %d, %d KV storages, page bytes %s, %d blocks, '
                 'min run %d, min new %d, margin %d; phase 2 (KV offload) %s',('OFF' if s.dead else 'ON'),s.want,PF_NF,s.rb,len(pg),
                 sorted(set(pg)),nb,s.minrun,MIN_NEW,MARGIN,
                 (f'ON: {s.n_cand} MLA layers x {s.kv_page} B/block, host <= {HOST_GB:g} GiB/rank x {s.world}, RAM floor {FLOOR_GB:g} GiB, '
                  f'no borrow at <= {s.max_graph} tokens') if s.kv_ok else 'off')
    def need(s,req):
        """blocks the running request still needs for its prompt (+ spec lookahead) beyond what it holds"""
        n=0
        for m in s.kvm.coordinator.single_type_managers:
            bs=m.block_size;have=len(m.req_to_blocks.get(req.request_id,()))
            n+=max(0,-(-(req.num_prompt_tokens+64)//bs)-have)
        return n
    def borrow(s,req):
        pool=s.pool;avail=pool.get_num_free_blocks()-s.need(req)-MARGIN
        if avail<s.minrun:return False
        free=[b for b in pool.blocks if b.ref_cnt==0 and not b.is_null and b.prev_free_block is not None]
        # contiguous id runs, hashless (never-hit) first, then cached ones; longest first (least carve waste)
        runs=[]
        for hashed in (False,True):
            ids=sorted(b.block_id for b in free if (b.block_hash is not None)==hashed)
            r0=None;prev=None
            for i in ids:
                if r0 is not None and i==prev+1:prev=i;continue
                if r0 is not None:runs.append((r0,prev+1,hashed))
                r0=prev=i
            if r0 is not None:runs.append((r0,prev+1,hashed))
        runs=[r for r in runs if r[1]-r[0]>=s.minrun];runs.sort(key=lambda r:(r[2],-(r[1]-r[0])))
        take=[];got=0;nb=0
        for b0,b1,_ in runs:
            if got>=s.want or nb>=avail:break
            n=min(b1-b0,avail-nb)
            if got+carve_count(n,s.pages,s.rb)>s.want:          # shortest prefix of the run that reaches the target
                lo,hi=1,n
                while lo<hi:
                    m=(lo+hi)//2
                    if got+carve_count(m,s.pages,s.rb)>=s.want:hi=m
                    else:lo=m+1
                n=lo
            c=carve_count(n,s.pages,s.rb)
            if c<=0:continue
            take.append((b0,b0+n));got+=c;nb+=n
        if not take:return False
        blk=[]
        for b0,b1 in take:
            for i in range(b0,b1):
                b=pool.blocks[i];pool.free_block_queue.remove(b)
                if pool.enable_caching:pool._maybe_evict_cached_block(b)
                b.ref_cnt=1;blk.append(b)
        s.ep+=1;s.mode=1;s.blocks=blk;s.runs=tuple(take);s.nslots=min(got,s.want);s.rid=req.request_id;s.n['borrows']+=1;s.n['blocks']+=nb;s.n['slots']+=min(got,s.want)
        log.info('NestQuant prefill-borrow: epoch %d req %s: %d blocks in %d runs -> %d slots (want %d), %d free left, %d prompt tokens to go',
                 s.ep,req.request_id,nb,len(take),min(got,s.want),s.want,pool.get_num_free_blocks(),req.num_prompt_tokens-req.num_computed_tokens)
        return True
    # --- phase 2
    def req_blocks(s,req):
        return [b.block_id for b in s.kvm.coordinator.single_type_managers[s.kv_gi].req_to_blocks.get(req.request_id,())]
    def kv_slots(s,n_off):
        return min(s.want,carve_count(s.nb,[s.kv_page]*max(0,n_off-2),s.rb))
    def kv_plan(s,req):
        """(n_off, rows, slots) phase 2 can do now for req, from the RAM available at this moment"""
        have=len(s.req_blocks(req));bs=s.kvm.coordinator.single_type_managers[s.kv_gi].block_size
        rows=max(have,-(-(req.num_prompt_tokens+64)//bs))+8
        per=rows*s.kv_page;ma=mem_avail();room=(ma-FLOOR_GB*2**30)/max(1,s.world)
        n_off=int(max(0.,min(HOST_GB*2**30,room))//per) if per else 0;n_off=min(n_off,s.n_cand)
        while n_off>3 and s.kv_slots(n_off-1)>=s.want:n_off-=1   # fewest layers that reach the target (less PCIe)
        return n_off,rows,(s.kv_slots(n_off) if n_off>=3 else 0),ma
    def kv_borrow(s,req,plan):
        n_off,rows,slots,ma=plan;pool=s.pool;ev=0
        if pool.enable_caching:                             # cached free blocks lose their GPU content
            for b in pool.blocks:
                if b.ref_cnt==0 and not b.is_null and pool._maybe_evict_cached_block(b):ev+=1
        s.ep+=1;s.mode=2;s.rid=req.request_id;s.runs=();s.nslots=slots;s.kv=dict(n_cand=s.n_cand,n_off=n_off,rows=rows,blocks=(),wb=())
        s.n['kv_borrows']+=1;s.n['slots']+=slots
        log.info('NestQuant prefill-borrow: epoch %d req %s: KV offload of %d/%d MLA layers (host %.1f GiB/rank, %d rows, MemAvailable %.1f GiB)'
                 ' -> %d slots (want %d), %d cached free blocks evicted, %d prompt tokens to go',s.ep,req.request_id,n_off,s.n_cand,
                 n_off*rows*s.kv_page/2**30,rows,ma/2**30,slots,s.want,ev,req.num_prompt_tokens-req.num_computed_tokens)
    def release(s,why):
        if not s.mode:return
        if s.mode==1:
            for b in s.blocks:b.ref_cnt=0
            s.pool.free_block_queue.prepend_n(s.blocks)        # hashless: first to be reused
        log.info('NestQuant prefill-borrow: epoch %d released (%s), %s, %d free',s.ep,why,
                 f'{len(s.blocks)} blocks back' if s.mode==1 else 'KV offload ends',s.pool.get_num_free_blocks())
        s.blocks=[];s.runs=();s.phase=None;s.rid=None;s.mode=0;s.kv=None;s.nslots=0
    def after(s,sched,out):
        rs=sched.running;r=rs[0] if len(rs)==1 else None
        ns=out.num_scheduled_tokens.get(r.request_id,0) if r is not None else 0
        pf=False;left=0;before=0
        if r is not None and ns>0:
            before=r.num_computed_tokens-ns;pf=before<r.num_prompt_tokens;left=r.num_prompt_tokens-r.num_computed_tokens
        if s.mode and (not pf or r.request_id!=s.rid):s.release('prefill done' if r is not None and r.request_id==s.rid else 'request change')
        if s.mode==2:
            ma=mem_avail()
            if ma<FLOOR_GB*2**30:s.release(f'MemAvailable {ma/2**30:.1f} GiB < floor');s.tried.discard(r.request_id)
            elif ns<=s.max_graph:s.release(f'{ns}-token step (graph size)')
            elif len(s.req_blocks(r))>s.kv['rows']:s.release('block list outgrew the host rows')
        first=r is not None and r.request_id not in s.seen
        if r is not None:
            s.seen.add(r.request_id)
            if len(s.seen)>4096:s.seen=set(list(s.seen)[-1024:])
        ok=pf and not s.dead and left>0 and r.num_prompt_tokens-before>=MIN_NEW and not os.path.exists(OFF)
        if ok and not s.mode and r.request_id not in s.tried:
            s.tried.add(r.request_id)
            if len(s.tried)>4096:s.tried=set(list(s.tried)[-1024:])
            s.borrow(r)
        if ok and s.kv_ok and not first and s.mode!=2 and r.request_id not in s.tried2 and ns>s.max_graph and \
           (s.mode==0 or s.nslots<s.want):
            plan=s.kv_plan(r)
            if plan[2]>max(s.nslots*1.1,s.nslots+64):
                s.tried2.add(r.request_id)
                if len(s.tried2)>4096:s.tried2=set(list(s.tried2)[-1024:])
                if s.mode:s.release('phase 2 takes over')
                s.kv_borrow(r,plan)
        if s.mode:s.phase='borrow' if left>0 else 'drain'
        if s.mode==2:
            bl=s.req_blocks(r);bs=s.kvm.coordinator.single_type_managers[s.kv_gi].block_size
            j0=before//bs;j1=min(len(bl),-(-(before+ns)//bs))
            s.kv['blocks']=tuple(bl);s.kv['wb']=tuple(bl[j0:j1])
        out.nq_pb=(s.ep,s.phase,s.runs,s.nslots,dict(s.kv) if s.mode==2 else None) if s.mode else None

_ST={}
def _state(sched):
    st=_ST.get(id(sched))
    if st is None:st=_ST[id(sched)]=State(sched);st.kvm._nq_pb=st
    return st

def patch_scheduler(mod):
    C=mod.Scheduler
    if getattr(C,'_nq_pb',False):return
    s0=C.schedule
    @functools.wraps(s0)
    def schedule(self,*a,**k):
        out=s0(self,*a,**k)
        try:
            st=_state(self)
            st.after(self,out)
        except Exception:
            log.exception('NestQuant prefill-borrow: scheduler hook failed, off')
            try:
                st=_ST.get(id(self))
                if st is not None:st.release('error');st.dead=True
            except Exception:pass
            out.nq_pb=None
        return out
    C.schedule=schedule;C._nq_pb=True
    log.info('NestQuant prefill-borrow: Scheduler.schedule patched')

def patch_kvm(mod):
    C=mod.KVCacheManager
    if getattr(C,'_nq_pb_cls',False):return
    a0=C.allocate_slots
    @functools.wraps(a0)
    def allocate_slots(self,*a,**k):
        r=a0(self,*a,**k)
        st=getattr(self,'_nq_pb',None)
        if r is None and st is not None and st.mode==1:     # block pressure: give the borrowed blocks back now, retry
            st.n['pressure']+=1;st.release('allocate_slots pressure');r=a0(self,*a,**k)
        return r
    C.allocate_slots=allocate_slots;C._nq_pb_cls=True

_TARGETS={'vllm.v1.core.sched.scheduler':patch_scheduler,'vllm.v1.core.kv_cache_manager':patch_kvm}

class _Finder:
    """meta-path hook: run the patch right after the target module executes (no early vllm import)"""
    def find_spec(self,name,path=None,target=None):
        if name not in _TARGETS:return None
        import importlib.machinery as M
        for f in sys.meta_path:
            if f is self or not hasattr(f,'find_spec'):continue
            spec=f.find_spec(name,path,target)
            if spec is not None:break
        else:return None
        ld=spec.loader;ex0=ld.exec_module
        def exec_module(m,_ex0=ex0,_n=name):
            _ex0(m)
            try:_TARGETS[_n](m)
            except Exception:log.exception('NestQuant prefill-borrow: patch of %s failed',_n)
        try:ld.exec_module=exec_module
        except Exception:return None
        return spec

def install():
    if not ON:return
    for n,f in _TARGETS.items():
        if n in sys.modules:f(sys.modules[n])
    if not any(isinstance(f,_Finder) for f in sys.meta_path):sys.meta_path.insert(0,_Finder())
