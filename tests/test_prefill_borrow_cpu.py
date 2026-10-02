"""nq-prefill (prefill-borrow) CPU tests, no GPU touched.
Part A (any python with torch + numpy): the real RankExecutor / OpLog / Follower / CoalescingFollower / Scheduler /
  lookahead _plan / nq_pb.PB on a fake engine + fake mailboxes (CPU tensors, addresses only), one leader + two followers,
  random interleaving of the streaming loops, the worker hooks and the kernel's per-layer mailbox apply, several
  borrow epochs. Checked at every forward of every rank:
    - after a reclaim hook returns, no expert row in any table (after the forward's mailbox apply) and no engine op in
      flight points into borrowed memory, and no row of the staged mailbox does either (the KV may be written now)
    - level-4 rows never share a slot address; normal-pool accounting (free + used == nslot)
  and at the end (quiescent): every follower's level-4 set == the leader's, Follower.check() ok.
Part B (needs vllm: run in the serve container): nq_pb_engine.State on vLLM's real BlockPool with a fake scheduler:
  borrow only free blocks (removed from the free queue, ref_cnt 1, hash evicted), reserve = prompt + margin kept,
  never borrowed below min-new or with several requests, release at the first decode step / request change /
  allocate_slots pressure, worker carve() of the runs == exactly n_slots aligned non-overlapping slots inside the runs.
run: CUDA_VISIBLE_DEVICES= /data/Jarrel/nqenv/bin/python tests/test_prefill_borrow_cpu.py
     docker exec -e CUDA_VISIBLE_DEVICES= glm53-nestquant /opt/vllm/.venv/bin/python /nq/tests/test_prefill_borrow_cpu.py B"""
import os,sys,types,random,tempfile,threading,collections
os.environ.update(NQ_PREFILL_BORROW='1',NQ_PREFILL_SLOTS='14',NQ_PB_MIN_NEW='1024',NQ_PB_MARGIN='4',NQ_HOSTLOOP='py',NQ_PREDICTOR='ema')
HERE=os.path.dirname(os.path.abspath(__file__));R=os.path.dirname(HERE)
sys.path[:0]=[R+'/streaming',R+'/sm120/serve',R+'/sm120']
import numpy as np,torch
# executor.py imports the CUDA kernel modules (JIT build): stand-ins, the test never runs a kernel
for _n in ('moe','p4rec'):
    if _n not in sys.modules:sys.modules[_n]=types.ModuleType(_n)
sys.modules['moe'].entry=None;sys.modules['p4rec'].row=None

class _T:
    """torch for the executor / nq_pb: everything real, devices are the CPU, no CUDA calls"""
    def __getattr__(s,n):return getattr(torch,n)
    @staticmethod
    def device(*a,**k):return torch.device('cpu')
    cuda=types.SimpleNamespace(current_device=lambda:0,synchronize=lambda *a,**k:None,Event=None)

NE=32;LAY=[3,4,5];RB=1<<16;SPL=8;NF=6;PF=14

# ------------------------------------------------------------------------------------------------ part A
class MB:
    def __init__(s):
        s.stage=torch.zeros(NE,2,dtype=torch.int64);s.seq=torch.zeros(NE,dtype=torch.int32)
        s.applied=torch.zeros(NE,dtype=torch.int32);s.applied_host=torch.zeros(NE,dtype=torch.int32);s.hseq=[0]*NE
        s.table=torch.zeros(NE,2,dtype=torch.int64);s.table[:,0]=2
    def apply(s):                       # the kernel: stage -> table where seq != applied
        ch=s.seq!=s.applied;s.table[ch]=s.stage[ch];s.applied.copy_(s.seq);s.applied_host.copy_(s.applied)

class FE:
    """fake nqstream engine: ops complete after a random latency; the stage row + seq land at completion"""
    def __init__(s,lay,rng):s.lay=lay;s.rng=rng;s.t=0;s.q=collections.OrderedDict();s.done=[];s.cancelled=set();s.auto=False
    def _ix(s,st):
        for L,m in s.lay.items():
            b=m.stage.data_ptr();sb=m.stage.stride(0)*8
            if b<=st<b+sb*NE:return L,(st-b)//sb
        raise AssertionError('bad stage address')
    def post(s,tag,st,row,sq,q):s.q[tag]=dict(k='d',ix=s._ix(st),row=row.clone(),q=q,t=s.t+1,started=True,dst=None)
    def upgrade(s,tag,rec,dst,st,row,sq,q):
        assert dst==int(row[1]),f"engine dst {dst:#x} != the slot the table row points at {int(row[1]):#x}"   # the SSD read lands where the table says
        s.q[tag]=dict(k='u',ix=s._ix(st),row=row.clone(),q=q,t=s.t+s.rng.randint(1,6),started=False,dst=dst)
    def cancel(s,tag):
        o=s.q.get(tag)
        if o is not None and not o['started']:s.cancelled.add(tag)
    def tick(s):
        s.t+=1
        for tag,o in list(s.q.items()):
            if tag in s.cancelled:del s.q[tag];s.cancelled.discard(tag);s.done.append((tag,0,-1e9,0.));continue
            if s.t>=o['t']-2:o['started']=True
            if s.t>=o['t']:
                L,E=o['ix'];m=s.lay[L];m.stage[E]=o['row'];m.seq[E]=o['q'];del s.q[tag];s.done.append((tag,0,1.,1e-3))
    def poll(s):
        if s.auto:s.tick()            # inside a reclaim the drive keeps going (the real engine runs on its own threads)
        d=s.done;s.done=[];return d
    def inflight_dst(s):return [o['dst'] for o in s.q.values() if o['dst'] is not None]
    def stats(s):return {}
    def close(s):pass

class RF:
    def __init__(s,eng):s.eng=eng;s.rb=RB;s.lay={}
    def rec(s,L,E):return L*NE+E
    def engine(s,*a,**k):return s.eng

def part_a(seed=0,epochs=4):
    import executor as EX,oplog as OL,scheduler as SC,nq_lookahead as LAH,nq_pb as PBM
    EX.torch=_T();PBM.torch=_T();LAH.NE=NE
    def mkrows(s,L,E):return torch.tensor([2,0],dtype=torch.int64),np.array([4,0],np.int64),np.array([0,1],np.int64)
    EX.RankExecutor._mkrows=mkrows
    rng=random.Random(seed);d=tempfile.mkdtemp();log0=OL.OpLog(d+'/oplog.bin',writer=True)
    fixed={L:[0,1] for L in LAY};dflt={L:list(range(2,2+NF)) for L in LAY}
    ranks=[]
    for r in range(3):
        lay={L:MB() for L in LAY};eng=FE(lay,random.Random(seed*7+r))
        X=EX.RankExecutor(RF(eng),{L:(None,lay[L],{E:None for E in range(NE)}) for L in LAY},SPL*len(LAY))
        S=(__import__('scheduler_tap').make_scheduler if os.environ.get('NQ_SCHED') else SC.Scheduler)(LAY,fixed,dflt,RB,NE=NE,n_float=NF,slots=SPL*len(LAY),cap_GBps=1e6,predictor='ema')
        init=[(L,E) for L in LAY for E in dflt[L]];[S.state.__setitem__((S.li[L],E),1) for L,E in init];X.apply(init,[],S)
        rt=types.SimpleNamespace(rank=r,dev=None,X=X,S=S,F=None,log=log0 if r==0 else None,cv=threading.Condition(),in_iter=False,pb_pause=0,lay=lay,eng=eng)
        if r:
            rt.log=OL.OpLog(d+'/oplog.bin',writer=False);rt.F=(OL.Follower if r==1 else OL.CoalescingFollower)(X,rt.log)
            rt.F.mark_busy(init);rt.F.enable_check(init);rt.F.gate=X
        rt.PB=PBM.PB(rt)
        # fake KV: two storages, page sizes of the real serve's MLA + a smaller one, far from the normal pool
        rt.PB.stor=[(1<<44,41984),((1<<44)+(1<<36),9216)]
        ranks.append(rt)
    la=types.SimpleNamespace(rt=None)
    stats=collections.Counter()
    def borrowed_ranges(runs):return [(b+b0*p,b+b1*p) for b,p in ranks[0].PB.stor for b0,b1 in runs]
    def inside(a,rs):return any(lo<=a<hi for lo,hi in rs)
    hist=[]                                            # all borrowed ranges ever (after reclaim nothing may point there)
    def leader_iter(prefill):
        rt=ranks[0];S=rt.S;X=rt.X
        X.poll(S)
        if prefill:
            for L in LAY:
                b=np.array([rng.random()**3 for _ in range(NE)])
                ups,downs,_,_=LAH.LA._plan(la,S,L,b,budget=4)
                if ups or downs:
                    if ups and X.xtag:ups=X.pool_tag(ups)
                    pin_check(S,downs);X.apply(ups,downs,S);log0.put(ups,downs)
        else:
            c=np.zeros((len(LAY),NE))
            for i in range(len(LAY)):
                for e in rng.sample(range(NE),8):c[i,e]+=rng.randint(1,4)
            ups,downs=S.step(c,16);pin_check(S,downs);ups=X.pool_tag(ups);X.apply(ups,downs,S);log0.put(ups,downs)
    def follower_iter(rt):rt.X.poll(rt.F);rt.F.step()
    def pin_check(S,downs):                            # NQ_PB_PROTECT: no pinned pre-borrow decode resident is downed during its borrow
        pin=ranks[0].PB.pin
        if pin is None:return
        stats['pinned_steps']+=1;bad=[(L,E) for L,E in downs if pin[S.li[L],E]];assert not bad,('protected expert downed',bad[:4])
    def forward_check(rt,cur_r,old):
        X=rt.X;seen={}
        for L,m in rt.lay.items():
            m.apply()
            for E in range(NE):
                lv,a=int(m.table[E,0]),int(m.table[E,1])
                if lv!=4:continue
                assert not inside(a,old) or inside(a,cur_r),f'rank {rt.rank}: L{L} E{E} table points into reclaimed KV: slot {X.slot_of.get((L,E))} wait {X.wait_apply.get((L,E))} hseq {m.hseq[E]} seq {int(m.seq[E])} app {int(m.applied[E])} stage {m.stage[E].tolist()} xep {X.xep} xst {dict(X.xst)} ops {[o for o in X.ops.values() if o[:2]==(L,E)]} pend {X.pend[:3]} xpend {X.xpend[:3]}'
                assert a not in seen,f'rank {rt.rank}: slot {a:#x} shared {seen[a]} ({L},{E})'
                seen[a]=(L,E)
                ok=X.slot0<=a<X.slot0+X.nslot*RB or inside(a,cur_r);assert ok,f'rank {rt.rank}: row outside every pool'
        for a in rt.eng.inflight_dst():assert not inside(a,old) or inside(a,cur_r),f'rank {rt.rank}: op in flight into reclaimed KV'
        n_norm=sum(1 for sl in X.slot_of.values() if sl<X.nslot);assert n_norm+len(X.free)==X.nslot,(n_norm,len(X.free))
        stats['x_slots_used']=max(stats['x_slots_used'],sum(1 for sl in X.slot_of.values() if sl>=X.nslot))
    ep=0;t=0
    plan=[]
    for k in range(epochs):plan+=[('dec',None)]*rng.randint(10,40)+[('pf','borrow')]*rng.randint(15,40)+[('pf','drain')]*rng.randint(1,4)
    plan+=[('dec',None)]*60
    cur=None;prev_phase=None
    for kind,phase in plan:
        t+=1
        if phase is not None and prev_phase is None:
            ep+=1;runs=((100+ep*3,180+ep*3),(300,300+rng.randint(20,90)));
            from nq_pb_engine import carve_count
            ns=min(sum(carve_count(b1-b0,[p for _,p in ranks[0].PB.stor],RB) for b0,b1 in runs),(PF-NF)*len(LAY))
            cur=borrowed_ranges(runs)
        so=types.SimpleNamespace(nq_pb=(ep,phase,runs,ns) if phase is not None else None)
        prev_phase=phase
        live=cur if so.nq_pb is not None else [];old=hist+(cur if so.nq_pb is None and cur else [])
        # one step: per rank hook -> forward (mailbox apply + checks); streaming loops interleave freely
        ev=[('hook',r) for r in range(3)]+[('loop',r) for r in range(3) for _ in range(rng.randint(1,3))]
        rng.shuffle(ev);q=[]
        for e in ev:
            q.append(e)
            if e[0]=='hook':q.append(('fwd',e[1]))
        for e,r in q:
            rt=ranks[r]
            if e=='hook':rt.eng.auto=True;rt.PB.on_sched(None,so);rt.eng.auto=False
            elif e=='fwd':forward_check(rt,live,old)
            else:
                if r==0:leader_iter(kind=='pf')
                else:follower_iter(rt)
        if so.nq_pb is None and cur is not None:hist+=cur;cur=None
        for rt in ranks:rt.eng.tick()
        stats['ticks']+=1
    # quiesce: decode with no new ops until nothing is in flight anywhere
    for _ in range(400):
        ranks[0].X.poll(ranks[0].S)
        for rt in ranks[1:]:follower_iter(rt)
        for rt in ranks:
            for m in rt.lay.values():m.apply()
            rt.eng.tick()
        if not any(rt.X.busy() or rt.eng.q for rt in ranks) and all(not rt.F.q and not rt.F.busy for rt in ranks[1:]):break
    S=ranks[0].S;lead={(L,E) for L in LAY for E in range(NE) if S.state[S.li[L],E]==2}
    assert not (S.state%2).any(),('leader ops still in flight',[(S.layers[i],int(e),int(S.state[i,e])) for i,e in zip(*np.nonzero(S.state%2))][:8],ranks[0].X.busy(),len(getattr(S,'todo',())),len(getattr(S,'doom',{})))
    for rt in ranks[1:]:
        assert rt.F.up==lead,(rt.rank,'F-only',sorted(rt.F.up-lead)[:8],'lead-only',sorted(lead-rt.F.up)[:8],[(k,int(S.state[S.li[k[0]],k[1]])) for k in sorted(rt.F.up^lead)[:8]],S.stats.get('shrink_evict'),rt.F.stats)
        r=rt.F.check();assert r is not None and r[0],(rt.rank,r,rt.F.check_msg,list(rt.F.q)[:6],sorted(rt.F.busy)[:6],rt.X.busy(),dict(list(rt.X.ops.items())[:4]),dict(list(rt.X.wait_apply.items())[:4]),rt.X.pend[:4],rt.X.xpend[:4],len(rt.eng.q))
    for rt in ranks:
        tab={(L,E) for L,m in rt.lay.items() for E in range(NE) if int(m.table[E,0])==4}
        assert tab==lead,(rt.rank,sorted(tab^lead)[:8])
        assert len(rt.X.free)==rt.X.nslot-len(lead),(rt.rank,len(rt.X.free),len(lead))
        assert not rt.X.xaddr and not rt.X.xep,rt.X.xaddr
    assert ranks[0].PB.pin is None and getattr(ranks[0].S,'pin',None) is None,'protect pin left after reclaim'
    assert (PBM.PROTECT>0)==(ranks[0].PB.n['protected']>0),(PBM.PROTECT,dict(ranks[0].PB.n))
    print(f'A seed {seed}: {epochs} epochs, {stats["ticks"]} steps, max borrowed slots used {stats["x_slots_used"]}, '
          f'leader xst {dict(ranks[0].X.xst)}, F1 {dict(ranks[1].X.xst)}, F2 {dict(ranks[2].X.xst)}, final level-4 floating {len(lead)}, protected {ranks[0].PB.n['protected']} over {stats['pinned_steps']} steps: OK')

# ------------------------------------------------------------------------------------------------ part B
def part_b():
    import json
    from vllm.v1.core.block_pool import BlockPool
    d=tempfile.mkdtemp();json.dump(dict(layers=[str(L) for L in LAY],rec_bytes=RB),open(d+'/rank0.json','w'));os.environ['NQ_REPACK']=d
    os.environ['NQ_LAYERS']=''
    import nq_pb_engine as PE,nq_pb as PBM
    PE.nf0=lambda:NF
    NB=600;BS=256;pages=[41984,9216]
    pool=BlockPool(NB,True,BS)
    class Mgr:
        block_size=BS
        def __init__(s):s.req_to_blocks={}
    mgr=Mgr()
    kvm=types.SimpleNamespace(block_pool=pool,coordinator=types.SimpleNamespace(single_type_managers=[mgr]))
    cfg=types.SimpleNamespace(num_blocks=NB,kv_cache_tensors=[types.SimpleNamespace(size=NB*p,block_stride=0) for p in pages])
    sched=types.SimpleNamespace(kv_cache_config=cfg,kv_cache_manager=kvm,running=[])
    st=PE.State(sched);assert not st.dead and st.want==(PF-NF)*len(LAY),(st.dead,st.want)
    class Req:
        def __init__(s,rid,n):s.request_id=rid;s.num_prompt_tokens=n;s.num_computed_tokens=0
    def step(r,ns,others=()):
        sched.running=[r,*others] if r is not None else list(others)
        out=types.SimpleNamespace(num_scheduled_tokens={r.request_id:ns} if r is not None else {})
        if r is not None:
            need=-(-(r.num_computed_tokens+ns)//BS)-len(mgr.req_to_blocks.get(r.request_id,[]))
            if need>0:mgr.req_to_blocks.setdefault(r.request_id,[]).extend(pool.get_new_blocks(need))
            r.num_computed_tokens+=ns
        st.after(sched,out);return out.nq_pb
    # some cached (hashed) free blocks + some in use by another request
    other=pool.get_new_blocks(40);pool.free_blocks(other[:20])
    f0=pool.get_num_free_blocks()
    r=Req('a',60000);pb=step(r,4096)
    assert pb is not None and pb[1]=='borrow',pb
    ep,ph,runs,ns=pb[:4];nblk=sum(b1-b0 for b0,b1 in runs)
    assert ns==min(st.want,sum(PE.carve_count(b1-b0,pages,RB) for b0,b1 in runs)),ns
    for b0,b1 in runs:
        for i in range(b0,b1):
            b=pool.blocks[i];assert b.ref_cnt==1 and b.prev_free_block is None and b.block_hash is None,i
    free=pool.get_num_free_blocks();assert free>=st.need(r)+PE.MARGIN,(free,st.need(r))
    # worker carve over the same runs: exactly ns slots, aligned, inside the runs, non-overlapping
    stor=[(1<<44,pages[0]),((1<<44)+(1<<36)+64,pages[1])]
    a=PBM.carve(stor,runs,RB,ns);assert len(a)==ns,(len(a),ns)
    iv=sorted((x,x+RB) for x in a)
    assert all(x%PE.ALIGN==0 for x in a) and all(iv[i][1]<=iv[i+1][0] for i in range(len(iv)-1))
    rs=[(b+b0*p,b+b1*p) for b,p in stor for b0,b1 in runs];assert all(any(lo<=x and x+RB<=hi for lo,hi in rs) for x in a)
    # the request's own allocations never get borrowed blocks; prompt finishes in chunks
    while r.num_computed_tokens<r.num_prompt_tokens:
        n=min(4096,r.num_prompt_tokens-r.num_computed_tokens);pb=step(r,n)
        for blk in mgr.req_to_blocks['a']:assert not any(b0<=blk.block_id<b1 for b0,b1 in runs)
        assert pb is not None and pb[0]==ep and pb[1]==('borrow' if r.num_computed_tokens<r.num_prompt_tokens else 'drain'),pb
    pb=step(r,1);assert pb is None and pool.get_num_free_blocks()+0>=0
    for b0,b1 in runs:
        for i in range(b0,b1):assert pool.blocks[i].ref_cnt==0 and pool.blocks[i].prev_free_block is not None
    # small prefill: never; two running: never; prefix-cached prompt: min-new counts uncached tokens only
    r2=Req('b',900);assert step(r2,900) is None
    r3=Req('c',50000);assert step(r3,4096,others=[r2]) is None
    r4=Req('d',5000);r4.num_computed_tokens=4096;assert step(r4,904) is None
    # pressure: allocate_slots None -> release -> retry
    r5=Req('e',30000);pb=step(r5,4096);assert pb is not None
    calls=[]
    class KVM:
        def allocate_slots(s,*a,**k):calls.append(1);return None if len(calls)==1 else 'ok'
    m=types.SimpleNamespace(KVCacheManager=KVM);PE.patch_kvm(m);k=KVM();k._nq_pb=st
    assert k.allocate_slots()=='ok' and len(calls)==2 and not st.blocks and st.n['pressure']==1
    print(f'B: borrowed {nblk} blocks in {len(runs)} runs -> {ns} slots (want {st.want}), free {f0} -> {free}, release/pressure/guards OK')

if __name__=='__main__':
    which=sys.argv[1] if len(sys.argv)>1 else 'A'
    if 'A' in which:
        for sd in range(int(os.environ.get('SEEDS','12'))):part_a(sd)
    if 'B' in which:part_b()
