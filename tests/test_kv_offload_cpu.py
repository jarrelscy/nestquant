"""nq-prefill phase 2 (NQ_PREFILL_KV_OFFLOAD=1) CPU tests, no GPU touched.
Part C (any python with torch): nq_pb.KVOff on a fake runner (MLAAttention layers with their own KV storages, plus an
  indexer cache, an MTP layer and two layers sharing a storage, which must stay resident), CUDA streams/events as no-ops
  (copies are synchronous on the CPU: this checks the data movement, not the stream ordering). A fake forward writes
  each step's tokens through get_attention_context (the wrapped one) and reads every earlier token of every layer back;
  between layers and steps the carved expert slots are overwritten with garbage (experts landing). Checked: every read
  sees the right KV, resident layers never move, after end() every original storage holds the right KV for every
  token, LMCache stores issued during the borrow are deferred and, replayed, read the right KV; slots lie inside the
  offloaded storages 2.. (never the staging ones, never a resident layer).
Part D (needs vllm: run in the serve container): nq_pb_engine.State phase 2 on vLLM's real BlockPool: phase 1 first,
  phase 2 takes over from the 2nd step when phase 1 can't reach the target, no blocks taken, cached free blocks
  evicted, kv dict (blocks / written blocks) right, n_off sized from MemAvailable for all ranks, released at a
  graph-size step / prefill end / request change / MemAvailable below the floor (then phase 1 may run again), never
  on a request's first step.
run: CUDA_VISIBLE_DEVICES= /data/Jarrel/nqenv/bin/python tests/test_kv_offload_cpu.py C
     docker exec -e CUDA_VISIBLE_DEVICES= -e PYTHONPATH= glm53-nestquant /opt/vllm/.venv/bin/python <copy>/tests/test_kv_offload_cpu.py D"""
import os,sys,types,random,ctypes,contextlib,tempfile,json
os.environ.update(NQ_PREFILL_BORROW='1',NQ_PREFILL_KV_OFFLOAD='1',NQ_PREFILL_SLOTS='14',NQ_PB_MIN_NEW='1024',NQ_PB_MARGIN='4',
                  NQ_HOSTLOOP='py',NQ_PREDICTOR='ema',NQ_PB_KV_HOST_GB='8',NQ_PB_RAM_FLOOR_GB='38')
HERE=os.path.dirname(os.path.abspath(__file__));R=os.path.dirname(HERE)
sys.path[:0]=[R+'/streaming',R+'/sm120/serve',R+'/sm120']
import torch

class _Ev:
    def record(s,*a):pass
class _St:
    def wait_event(s,e):pass
    def wait_stream(s,o):pass
_cuda=types.SimpleNamespace(Event=_Ev,Stream=lambda *a,**k:_St(),current_stream=lambda *a:_St(),stream=lambda st:contextlib.nullcontext(),
                            synchronize=lambda *a:None,is_available=lambda:False)
class _T:
    def __getattr__(s,n):return getattr(torch,n)
    cuda=_cuda

def part_c(seed=0):
    import nq_pb as PBM
    PBM.torch=_T()
    if 'vllm' not in sys.modules:                       # stub vllm attention module for patch_gac
        for m in ('vllm','vllm.model_executor','vllm.model_executor.layers','vllm.model_executor.layers.attention'):
            sys.modules.setdefault(m,types.ModuleType(m))
        A=types.ModuleType('vllm.model_executor.layers.attention.attention');A.get_attention_context=lambda n:None
        sys.modules[A.__name__]=A;sys.modules['vllm.model_executor.layers.attention'].attention=A
    rnd=random.Random(seed)
    NB=48;BS=4;D=64;PG=BS*D;RB=1536;NL=10
    class MLAAttention:
        def __init__(s,kv):s.kv_cache=kv
    class Other:
        def __init__(s,kv):s.kv_cache=kv
    ctx={};orig={}
    for i in range(NL):
        n=f'model.layers.{i}.self_attn.attn';ctx[n]=MLAAttention(torch.zeros(NB,BS,D,dtype=torch.uint8));orig[n]=ctx[n].kv_cache
    sh=torch.zeros(NB,BS,D,dtype=torch.uint8)                 # two "layers" on one storage: never offloaded
    ctx['model.layers.20.self_attn.attn']=MLAAttention(sh);ctx['model.layers.21.self_attn.attn']=MLAAttention(sh)
    ctx['model.layers.3.self_attn.indexer.k_cache']=Other(torch.zeros(NB,BS,16,dtype=torch.uint8))
    ctx[f'model.layers.{NL+30}.self_attn.attn']=MLAAttention(torch.zeros(NB,BS,D,dtype=torch.uint8))   # MTP layer (idx >= n layers)
    nL=NL+30
    for i in (20,21):ctx[f'model.layers.{i}.self_attn.attn'].kv_cache=sh
    res_names=[n for n in ctx if not n.startswith('model.layers.') or int(n.split('.')[2])>=NL or 'indexer' in n]
    res0={n:ctx[n].kv_cache.clone() for n in res_names}
    runner=types.SimpleNamespace(compilation_config=types.SimpleNamespace(static_forward_context=ctx),kv_cache_config=types.SimpleNamespace(num_blocks=NB),
                                 model_config=types.SimpleNamespace(hf_text_config=types.SimpleNamespace(num_hidden_layers=nL)))
    rt=types.SimpleNamespace(dev='cpu',rank=0,PB=None)
    ko=PBM.KVOff(rt);rt.PB=types.SimpleNamespace(KO=ko)
    stores=[]
    class Eng:
        def store(s,**k):
            # LMCache reads the registered (original) tensors at store time
            stores.append(('now' if 'store' not in s.__dict__ else 'x',k['tag']))
            for n,kv in zip(k['names'],k['kvcaches']):
                for pos,slot in zip(k['tag'],k['slot_mapping']):
                    assert int(kv.view(-1,D)[slot][0])==val(n,pos),('LMCache store read wrong KV',n,pos)
            s.done.append(k['tag'])
    eng=Eng();eng.done=[]
    ko._lmcache=lambda:(eng,True)
    def val(n,pos):return (hash(n)%97*31+pos*7+1)%251
    def gac(n):
        if ko.on:ko.touch(n)
        return ctx[n].kv_cache
    cand=sorted([n for n in ctx if n not in res_names and n not in ('model.layers.20.self_attn.attn','model.layers.21.self_attn.attn')],key=lambda n:int(n.split('.')[2]))
    assert [c[1] for c in ko.candidates(runner)]==cand,[c[1] for c in ko.candidates(runner)]
    free=list(range(NB));rnd.shuffle(free);free=sorted(free[:NB//2])+free[NB//2:]   # partly contiguous ids
    blocks=[];prompt=150;pos=0;slots=[];step_i=0;kvd=None;n_off=rnd.choice([3,5,NL])
    order=lambda:[n for n in ctx if n in cand or n in res_names]
    def clobber():
        for a in slots:ctypes.memset(a,rnd.randrange(256),RB)
    while pos<prompt:
        ns=min(rnd.choice([7,13,24,40]),prompt-pos)
        while len(blocks)*BS<pos+ns:blocks.append(free.pop(0))
        wb=blocks[pos//BS:-(-(pos+ns)//BS)]
        if step_i==1:                                   # phase 2 from the 2nd step
            kvd=dict(n_cand=len(cand),n_off=n_off,rows=-(-prompt//BS)+2,blocks=tuple(blocks),wb=tuple(wb))
            slots=ko.begin(runner,kvd,RB,1000);assert ko.on
            ko.step(kvd)
            offn=cand[:n_off];lo=[(orig[n].data_ptr(),orig[n].data_ptr()+NB*PG) for n in offn[2:]]
            assert len(slots)==sum((NB*PG-4096)//RB for _ in offn[2:]),len(slots)
            assert all(any(a>=l and a+RB<=h for l,h in lo) for a in slots),'slot outside offloaded storages 2..'
            for n in res_names:assert ctx[n].kv_cache is not None
        elif ko.on:
            kvd=dict(kvd,blocks=tuple(blocks),wb=tuple(wb));ko.step(kvd)
        clobber()
        names=order()
        if step_i==3:names=names[::-1]                      # out of order once: the miss path
        for n in names:
            if n in res_names:assert torch.equal(ctx[n].kv_cache,res0[n]),'resident layer changed';continue
            kv=gac(n)
            if ko.on and n in cand[:n_off]:assert kv.data_ptr() in (orig[cand[0]].data_ptr(),orig[cand[1]].data_ptr()),'not on staging'
            f=kv.view(-1,D)
            for p in range(pos,pos+ns):f[blocks[p//BS]*BS+p%BS]=val(n,p)
            for p in range(pos+ns):assert int(f[blocks[p//BS]*BS+p%BS][0])==val(n,p),('read wrong KV',n,p,step_i)
            if ko.on:clobber()
        sm=[blocks[p//BS]*BS+p%BS for p in range(pos,pos+ns)]
        eng.store(kvcaches=[orig[n] for n in cand],names=cand,slot_mapping=sm,tag=list(range(pos,pos+ns))) if not ko.on else \
            eng.store(kvcaches=[orig[n] for n in cand],names=cand,slot_mapping=sm,tag=list(range(pos,pos+ns)))
        pos+=ns;step_i+=1
    clobber();nd=len(ko.deferred);ko.end();assert not ko.on and ko.host is None
    for n in cand:
        assert ctx[n].kv_cache is orig[n],'original tensor not re-bound'
        f=orig[n].view(-1,D)
        for p in range(prompt):assert int(f[blocks[p//BS]*BS+p%BS][0])==val(n,p),('restored KV wrong',n,p)
    assert sorted(p for t in eng.done for p in t)==list(range(prompt)),'LMCache stores lost / duplicated'
    assert 'store' not in eng.__dict__
    print(f'C seed {seed}: n_off {n_off}/{len(cand)}, {len(slots)} slots, {step_i} steps, {nd} deferred stores, {dict(ko.n)} OK')

def part_d():
    from vllm.v1.core.block_pool import BlockPool
    LAY=[3,4,5];RBb=1<<16;NF=6
    d=tempfile.mkdtemp();json.dump(dict(layers=[str(L) for L in LAY],rec_bytes=RBb),open(d+'/rank0.json','w'));os.environ['NQ_REPACK']=d
    os.environ['NQ_LAYERS']=''
    import nq_pb_engine as PE
    PE.nf0=lambda:NF;PE.PF_NF=2000                  # want > what phase 1 can reach
    NB=2000;BS=256;NL=12;PG=41984;IPG=9216
    pool=BlockPool(NB,True,BS)
    class Mgr:
        block_size=BS
        def __init__(s):s.req_to_blocks={}
    mla=Mgr();idx=Mgr()
    kvm=types.SimpleNamespace(block_pool=pool,coordinator=types.SimpleNamespace(single_type_managers=[mla,idx]))
    names=[f'model.layers.{i}.self_attn.attn' for i in range(NL)]+[f'model.layers.{NL}.self_attn.attn']
    inames=[f'model.layers.{i}.self_attn.indexer.k_cache' for i in range(0,NL,4)]
    T=lambda p,n:types.SimpleNamespace(size=NB*p,block_stride=0,shared_by=[n])
    cfg=types.SimpleNamespace(num_blocks=NB,kv_cache_tensors=[T(PG,n) for n in names]+[T(IPG,n) for n in inames],
                              kv_cache_groups=[types.SimpleNamespace(layer_names=names),types.SimpleNamespace(layer_names=inames)])
    vc=types.SimpleNamespace(parallel_config=types.SimpleNamespace(world_size=4),compilation_config=types.SimpleNamespace(max_cudagraph_capture_size=32,cudagraph_capture_sizes=[1,32]),
                             model_config=types.SimpleNamespace(hf_text_config=types.SimpleNamespace(num_hidden_layers=NL)))
    sched=types.SimpleNamespace(kv_cache_config=cfg,kv_cache_manager=kvm,running=[],vllm_config=vc)
    MA=[200<<30];PE.mem_avail=lambda:MA[0]
    st=PE.State(sched);assert st.kv_ok and st.n_cand==NL and st.kv_page==PG and st.kv_gi==0,(st.kv_ok,st.n_cand,st.kv_page)
    class Req:
        def __init__(s,rid,n):s.request_id=rid;s.num_prompt_tokens=n;s.num_computed_tokens=0
    def step(r,ns):
        sched.running=[r] if r is not None else []
        out=types.SimpleNamespace(num_scheduled_tokens={r.request_id:ns} if r is not None else {})
        if r is not None:
            for m in (mla,idx):
                need=-(-(r.num_computed_tokens+ns)//BS)-len(m.req_to_blocks.get(r.request_id,[]))
                if need>0:m.req_to_blocks.setdefault(r.request_id,[]).extend(pool.get_new_blocks(need))
            r.num_computed_tokens+=ns
        st.after(sched,out);return out.nq_pb
    def done(r):
        for m in (mla,idx):pool.free_blocks(m.req_to_blocks.pop(r.request_id,[])[::-1])
    # fill the pool: a big other request's cached blocks, so phase 1 can't reach the target
    other=Req('o',0);ob=pool.get_new_blocks(1500)
    from vllm.v1.core.kv_cache_utils import BlockHash,make_block_hash_with_group_id
    for i,b in enumerate(ob[:300]):
        k=make_block_hash_with_group_id(BlockHash(i.to_bytes(8,'little')),0);b.set_block_hash(k);pool.cached_block_hash_to_block.insert(k,b)
    pool.free_blocks(ob[:300][::-1])
    assert sum(1 for b in pool.blocks if b.ref_cnt==0 and b.block_hash is not None)==300
    r=Req('a',60000);pb=step(r,4096)
    assert pb is not None and pb[4] is None,'phase 1 first';ns1=pb[3]
    assert ns1<st.want,(ns1,st.want)
    pb=step(r,4096);assert pb is not None and pb[4] is not None and pb[0]==2,'phase 2 takes over at step 2'
    kv=pb[4];cc=lambda n:PE.carve_count(NB,[PG]*(n-2),RBb);assert not st.blocks and kv['n_cand']==NL and cc(kv['n_off'])>=st.want>cc(kv['n_off']-1) and pb[3]==st.want,(kv['n_off'],pb[3])
    bl=[b.block_id for b in mla.req_to_blocks['a']];assert list(kv['blocks'])==bl and list(kv['wb'])==bl[16:32],(kv['wb'],bl[16:32])
    assert all(b.block_hash is None for b in pool.blocks if b.ref_cnt==0),'cached free block not evicted'
    assert all(pool.cached_block_hash_to_block.get_one_block(make_block_hash_with_group_id(BlockHash(i.to_bytes(8,'little')),0)) is None for i in range(300))
    rows0=kv['rows'];assert rows0>=-(-60000//BS)
    while r.num_computed_tokens<60000-20:
        n=min(4096,60000-20-r.num_computed_tokens);pb=step(r,n)
        assert pb is not None and pb[4] is not None and pb[0]==2 and pb[4]['rows']==rows0
    pb=step(r,20);assert pb is None,'graph-size step releases'
    pb=step(r,1);assert pb is None
    # first step of a request never does phase 2; MemAvailable sizing for 4 ranks; floor release -> phase 1 again
    r2rows=-(-(90000+64)//BS)+8;MA[0]=int((38+4*NL*0.5*r2rows*PG/2**30)*2**30)+(1<<20)   # room for half the layers on 4 ranks
    done(r);r2=Req('b',90000);pb=step(r2,4096);assert pb is None or pb[4] is None
    pb=step(r2,4096);assert pb is not None and pb[4] is not None,'phase 2 (r2)'
    n_off=pb[4]['n_off'];assert abs(n_off-NL//2)<=1 and 4*n_off*pb[4]['rows']*PG<=MA[0]-38*2**30,(n_off,NL)
    MA[0]=30<<30;pb=step(r2,4096);assert pb is None or pb[4] is None,'floor release'
    assert 'b' in st.tried2
    MA[0]=200<<30;pb=step(r2,4096);assert pb is None or pb[4] is None,'phase 2 not retried for that request'
    done(r2);r3=Req('c',90000);step(r3,4096);pb=step(r3,4096);assert pb[4] is not None
    done(r3);pb=step(Req('d',90000),4096);assert pb is None or pb[4] is None,'request change releases'
    print(f'D: phase 1 {ns1} slots -> phase 2 {PE.carve_count(NB,[PG]*(NL-2),RBb)} (want {st.want}), RAM-sized n_off {n_off}/{NL}, releases OK, {st.n}')

if __name__=='__main__':
    which=sys.argv[1] if len(sys.argv)>1 else 'C'
    if 'C' in which:
        for sd in range(int(os.environ.get('SEEDS','8'))):part_c(sd)
    if 'D' in which:part_d()
