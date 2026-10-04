"""NQ_LMPF slot borrow (nq_slotborrow.py + nq_lmpf.py borrow paths) CPU tests, no GPU touched.
The real RankExecutor / Scheduler / TapScheduler / OpLog / Follower / CoalescingFollower on fake mailboxes + a fake
nqstream engine (CPU tensors), TP ranks simulated by threads with a barrier all-reduce (MAX).
  1 span: lend / pick_span / span views (dtype, shape, alignment, inside the span)
  2 ring: own=False ring addresses = borrowed base, table rows z + m * addr, no memory -> issue refused
  3 invalidation: after a borrow no live table row (any rank) points into the span, evicted rows level 2, relocated
    records byte-identical at their new slot, S.state[U] == 0, tap doom / todo purged, S.slots reduced
  4 lent slots: _rel / cancel / failed reads never put a lent slot back in free, apply never takes one
  5 union: 3 ranks with different slot maps borrow the same U; streaming during the borrow never targets the span;
    after the return every follower's level-4 set == the leader's (Follower.check), accounting exact
  6 return: unlend restores free + S.slots + want = D0; a second borrow / return cycle is identical
  7 abort: exception in an LMPF step with a borrow held -> the next non-LMPF step returns it exactly once
  8 size plan: borrow sizing fits the budget (ring and all), BORROW=0 sizing unchanged; arena column plan + placement
run: docker run --rm -e CUDA_VISIBLE_DEVICES= -v <repo>:/nq <serve image> /opt/vllm/.venv/bin/python /nq/tests/test_slotborrow_cpu.py"""
import inspect
import os,sys,types,random,tempfile,threading,collections
os.environ.update(NQ_HOSTLOOP='py',NQ_PREDICTOR='ema',NQ_LMPF='1')
HERE=os.path.dirname(os.path.abspath(__file__));R=os.path.dirname(HERE)
sys.path[:0]=[R+'/streaming',R+'/sm120/serve',R+'/sm120']
import numpy as np,torch
for _n in ('moe','p4rec'):
    if _n not in sys.modules:sys.modules[_n]=types.ModuleType(_n)
sys.modules['moe'].entry=None;sys.modules['p4rec'].row=None
# CPU stand-ins for the few CUDA calls on the borrow paths
torch.cuda.synchronize=lambda *a,**k:None
torch.cuda.current_stream=lambda *a,**k:types.SimpleNamespace(synchronize=lambda:None)
class _T:
    def __getattr__(s,n):return getattr(torch,n)
    @staticmethod
    def device(*a,**k):return torch.device('cpu')
    cuda=types.SimpleNamespace(current_device=lambda:0,synchronize=lambda *a,**k:None,Event=None)
import executor as EX,oplog as OL,scheduler as SC,scheduler_tap as STP,nq_slotborrow as SB,nq_lmpf as LP
EX.torch=_T()
NE=32;LAY=[3,4,5];RB=4096;SPL=8;NF=6;NSLOT=SPL*len(LAY)

class MB:
    def __init__(s):
        s.stage=torch.zeros(NE,2,dtype=torch.int64);s.seq=torch.zeros(NE,dtype=torch.int32)
        s.applied=torch.zeros(NE,dtype=torch.int32);s.applied_host=torch.zeros(NE,dtype=torch.int32);s.hseq=[0]*NE
        s.table=torch.zeros(NE,2,dtype=torch.int64);s.table[:,0]=2
    def apply(s):
        ch=s.seq!=s.applied;s.table[ch]=s.stage[ch];s.applied.copy_(s.seq);s.applied_host.copy_(s.applied)
class FE:
    """fake engine: ops land after a random latency (stage row + seq written at completion); the record bytes are
    written to the destination (a signature: rec id) so relocations can be checked"""
    def __init__(s,lay,rng,X=None):s.lay=lay;s.rng=rng;s.t=0;s.q=collections.OrderedDict();s.done=[];s.cancelled=set();s.X=X;s.fail=set()
    def _ix(s,st):
        for L,m in s.lay.items():
            b=m.stage.data_ptr();sb=m.stage.stride(0)*8
            if b<=st<b+sb*NE:return L,(st-b)//sb
        raise AssertionError('bad stage address')
    def post(s,tag,st,row,sq,q):s.q[tag]=dict(k='d',ix=s._ix(st),row=row.clone(),q=q,t=s.t+1,started=True,dst=None,rec=None)
    def upgrade(s,tag,rec,dst,st,row,sq,q):
        assert dst==int(row[1])
        s.q[tag]=dict(k='u',ix=s._ix(st),row=row.clone(),q=q,t=s.t+s.rng.randint(1,5),started=False,dst=dst,rec=rec)
    def cancel(s,tag):
        o=s.q.get(tag)
        if o is not None and not o['started']:s.cancelled.add(tag)
    def tick(s):
        s.t+=1
        for tag,o in list(s.q.items()):
            if tag in s.cancelled:del s.q[tag];s.cancelled.discard(tag);s.done.append((tag,0,-1e9,0.));continue
            if s.t>=o['t']-1:o['started']=True
            if s.t>=o['t']:
                L,E=o['ix'];m=s.lay[L];del s.q[tag]
                if o['rec'] in s.fail:s.done.append((tag,0,-1.,1e-3));continue
                m.stage[E]=o['row'];m.seq[E]=o['q'];s.done.append((tag,0,1.,1e-3))
                if o['k']=='u' and s.X is not None:
                    X=s.X;sl=(o['dst']-X.slot0)//X.rb;X.slots[sl,:8]=torch.tensor(list(int(o['rec']).to_bytes(8,'little')),dtype=torch.uint8)
    def poll(s):d=s.done;s.done=[];return d
    def inflight_dst(s):return [o['dst'] for o in s.q.values() if o['dst'] is not None]
    def stats(s):return {}
    def close(s):pass
class RF:
    def __init__(s,eng):s.eng=eng;s.rb=RB;s.lay={}
    def rec(s,L,E):return L*NE+E
    def engine(s,*a,**k):return s.eng
def mkrows(s,L,E):return torch.tensor([2,0],dtype=torch.int64),np.array([4,0],np.int64),np.array([0,1],np.int64)
EX.RankExecutor._mkrows=mkrows
def sig(X,sl):return int.from_bytes(bytes(X.slots[sl,:8].tolist()),'little')

class Cluster:
    def __init__(s,seed=0,tap=False,ranks=3):
        s.rng=random.Random(seed);d=tempfile.mkdtemp();log0=OL.OpLog(d+'/oplog.bin',writer=True)
        fixed={L:[0,1] for L in LAY};dflt={L:list(range(2,2+NF)) for L in LAY};s.R=[]
        for r in range(ranks):
            lay={L:MB() for L in LAY};eng=FE(lay,random.Random(seed*7+r))
            X=EX.RankExecutor(RF(eng),{L:(None,lay[L],{E:None for E in range(NE)}) for L in LAY},NSLOT);eng.X=X
            C=STP.TapScheduler if tap else SC.Scheduler;kw=dict(NE=NE,n_float=NF,slots=NSLOT,cap_GBps=1e6,predictor='ema')
            if 'cap_GBps' not in inspect.signature(C.__init__).parameters:kw.pop('cap_GBps');kw['predictor']=None if tap else 'ema'   # spark-b175 scheduler: object predictor, no cap
            S=C(LAY,fixed,dflt,RB,**kw)
            init=[(L,E) for L in LAY for E in dflt[L]]
            for L,E in init:S.state[S.li[L],E]=1
            X.apply(init,[],S)
            rt=types.SimpleNamespace(rank=r,dev=torch.device('cpu'),X=X,S=S,F=None,log=log0 if r==0 else None,cv=threading.Condition(),
                                     in_iter=False,ncap=0,lm_hold=0,mbs=lay,lay=lay,eng=eng)
            if r:
                rt.log=OL.OpLog(d+'/oplog.bin',writer=False);rt.F=(OL.Follower if r==1 else OL.CoalescingFollower)(X,rt.log)
                rt.F.mark_busy(init);rt.F.enable_check(init)
            s.R.append(rt)
    def it(s,rt,issue=True,step=True):
        """one streaming-loop iteration of rank rt (issue False = executor fenced: rt.ncap)"""
        rt.eng.tick()
        if rt.F is not None:rt.X.poll(rt.F,issue=issue);rt.F.step(issue=issue)
        else:
            rt.X.poll(rt.S,issue=issue)
            if issue and step:
                c=np.zeros((len(LAY),NE))
                for i in range(len(LAY)):
                    for e in s.rng.sample(range(NE),10):c[i,e]+=s.rng.randint(1,5)
                ups,downs=rt.S.step(c,16);rt.X.apply(ups,downs,rt.S);rt.log.put(ups,downs)
        for m in rt.mbs.values():m.apply()            # the forward's mailbox apply
    def run(s,n,issue=True):
        for _ in range(n):
            for rt in s.rng.sample(s.R,len(s.R)):
                if s.rng.random()<.8:s.it(rt,issue)
    def fence(s,limit=500):
        """rt.ncap: no new ops, in-flight ones finish"""
        for _ in range(limit):
            for rt in s.R:s.it(rt,issue=False)
            if not any(rt.X.ops for rt in s.R):return
        raise AssertionError('fence did not drain')
    def quiesce(s,limit=3000):
        for _ in range(limit):
            for rt in s.R:s.it(rt,issue=True,step=False)
            if all(not rt.X.ops and not rt.X.pend and not rt.X.wait_apply and (rt.F is None or not (rt.F.q or rt.F.busy)) for rt in s.R):
                for rt in s.R:s.it(rt,issue=True,step=False)
                if all(not rt.X.ops and (rt.F is None or not rt.F.q) for rt in s.R):return
        raise AssertionError('not quiescent')
    def borrow(s,n,cant=(),margin=2):
        """the collective on every rank (threads + barrier all-reduce MAX)"""
        N=len(s.R);b=threading.Barrier(N);buf=[None]*N;out=[None];res=[None]*N;err=[]
        def mk(r):
            def ar(v):
                buf[r]=np.asarray(v).copy()
                if b.wait()==0:out[0]=np.maximum.reduce(buf)
                b.wait();o=out[0].copy();b.wait();return o
            return ar
        def go(r):
            try:res[r]=SB.borrow(s.R[r],n,mk(r),cant=r in cant,margin=margin)
            except Exception as e:
                import traceback;traceback.print_exc();err.append(e);b.abort()
        th=[threading.Thread(target=go,args=(r,)) for r in range(N)]
        for t in th:t.start()
        for t in th:t.join()
        assert not err,err
        return res
    def check(s,span=None):
        """accounting + no live row into the span + row/slot agreement"""
        for rt in s.R:
            X=rt.X;n_norm=sum(1 for sl in X.slot_of.values() if sl<X.nslot)
            assert n_norm+len(X.free)+len(X.lent)==X.nslot,(rt.rank,n_norm,len(X.free),len(X.lent))
            assert not (set(X.free)&X.lent) and len(set(X.free))==len(X.free)
            assert not (set(X.slot_of.values())&X.lent),'a lent slot holds an expert'
            for a in rt.eng.inflight_dst():assert (a-X.slot0)//X.rb not in X.lent,'op in flight into a lent slot'
            for L,m in rt.mbs.items():
                for E in range(NE):
                    if int(m.table[E,0])==4:
                        sl=(int(m.table[E,1])-X.slot0)//X.rb;assert sl not in X.lent,(rt.rank,L,E,sl)
                        if (L,E) in X.slot_of and not X.wait_apply.get((L,E)):assert X.slot_of[L,E]==sl
            if span is not None:
                a,n=span
                for L,m in rt.mbs.items():
                    lv4=m.table[:,0]==4;ad=m.table[:,1]
                    assert not bool(((ad>=X.addr(a))&(ad<X.addr(a+n))&lv4).any()),'live row into the span'

def test_span_lend():
    c=Cluster(1,ranks=1);c.run(20);c.fence();X=c.R[0].X
    a=SB.pick_span(X,6);occ=[k for k,sl in X.slot_of.items() if a<=sl<a+6]
    occ_min=min(sum(1 for sl in X.slot_of.values() if b<=sl<b+6) for b in range(X.nslot-5))
    assert len(occ)==occ_min
    try:
        if occ:X.lend(a,6);assert False,'lend of an occupied span'
    except AssertionError as e:assert 'span holds' in str(e)
    sp=[sl for sl in range(X.nslot) if sl not in X.slot_of.values()][:2]
    if len(sp)==2 and sp[1]==sp[0]+1:
        X.lend(sp[0],2);assert X.lent=={sp[0],sp[0]+1} and not (set(X.free)&X.lent)
        v=X.span_view(sp[0],2);assert v.numel()==2*RB and v.data_ptr()==X.addr(sp[0])
        X.unlend()
    # window views in a borrowed span (bw_bind, eager 'all')
    lm=LP.LM(types.SimpleNamespace(rank=0,X=X,dev=torch.device('cpu')));lm.bw='all';lm.C=1;lm.cg=None;lm.H=64;lm.esz=2;lm.dtype=torch.bfloat16
    lm.tib=torch.zeros(4,8,dtype=torch.int32);lm.tb=32;lm.per_tok=2*64*2+32
    n=lm.bw_need(50);fr=sorted(X.free)
    run=[b for b in range(X.nslot-n+1) if all(x in fr for x in range(b,b+n))]
    if run:
        X.lend(run[0],n);lm.bwi=dict(a=run[0],n=n);lm.bw_bind(50)
        lo,hi=X.addr(run[0]),X.addr(run[0]+n);bufs=[lm.hbuf,lm.rbuf,lm.stash]
        assert lm.hbuf.shape==(50,64) and lm.hbuf.dtype==torch.bfloat16 and lm.stash.shape==(50,8) and lm.stash.dtype==torch.int32
        ivs=sorted((t.data_ptr(),t.data_ptr()+t.numel()*t.element_size()) for t in bufs)
        assert all(lo+2*lm.C*RB<=x0 and x1<=hi and x0%LP.AL==0 for x0,x1 in ivs) and all(ivs[i][1]<=ivs[i+1][0] for i in range(2))
        X.unlend()

def test_ring_addrs():
    eng=types.SimpleNamespace(q=[],upgrade=lambda *a:eng.q.append(a),poll=lambda:[])
    rows={(3,E):(None,np.array([4,0]),np.array([0,1])) for E in range(8)}
    Rg=LP.Ring(eng,64,4,torch.device('cpu'),rows,lambda L,E:E,own=False)
    assert Rg.buf is None
    try:Rg.issue(0,3,[1]);assert False
    except AssertionError as e:assert 'no memory' in str(e)
    Rg.set_base(1<<20);Rg.issue(1,3,[5,6])
    assert [a[2] for a in eng.q]==[(1<<20)+(4+0)*64,(1<<20)+(4+1)*64] and int(eng.q[0][4][1])==Rg.addr(1,0)
    base=torch.zeros(LP.NE,LP.ROW_W,dtype=torch.int64);base[:,0]=2
    r=lambda a:torch.tensor([4,a]+[0]*(LP.ROW_W-2));wt=Rg.table(1,base,[(6,r(Rg.addr(1,1))),(5,r(Rg.addr(1,0)))])
    assert int(wt[5,1])==Rg.addr(1,0) and int(wt[6,1])==Rg.addr(1,1) and int(wt[0,0])==2

def _borrow_check(c,n,res):
    a0=[r['a'] for r in res];U0={r['U'] for r in res}
    assert len(U0)==1,('ranks disagree on |U|',U0)
    for rt,r in zip(c.R,res):
        X=rt.X;assert X.lent==set(range(r['a'],r['a']+n))
        for (L,E),sl in X.slot_of.items():
            if sl<X.nslot:assert sig(X,sl)==L*NE+E,('relocated record bytes',rt.rank,L,E,sl,sig(X,sl))
    c.check()
    for rt,r in zip(c.R,res):c.check.__func__(types.SimpleNamespace(R=[rt]),span=(r['a'],n))
    return a0

def test_invalidation_union_return(seed=3,tap=False):
    c=Cluster(seed,tap=tap);c.run(40)
    for rt in c.R:                    # followers lag: some of the log not replayed yet
        if rt.F is not None:
            for _ in range(c.rng.randint(0,3)):c.it(rt)
    c.fence()
    S=c.R[0].S;st0=S.state.copy();want0=(np.isin(S.state,(1,2))&~S.fixed)
    n=14;res=c.borrow(n)
    assert all(r is not None for r in res)
    _borrow_check(c,n,res)
    assert S.slots==NSLOT-n and int((S.state==2).sum())+n<=NSLOT
    ev=[(L,E) for L in LAY for E in range(NE) if st0[S.li[L],E]==2 and S.state[S.li[L],E]==0]
    assert len(ev)>=1 and all(int(c.R[0].mbs[L].table[E,0])==2 for L,E in ev)
    if tap:assert not any(i==S.li[L] and e==E for (i,e) in S.doom for L,E in ev) and not any((S.layers[i],e) in set(ev) for i,e in S.todo)
    # streaming continues with the span lent (op mode: fence released)
    for _ in range(30):
        c.run(1);c.check()
    c.fence()
    for rt,r in zip(c.R,res):
        nb=SB.give_back(rt,r);assert nb==n and not rt.X.lent
    assert S.slots==NSLOT
    if not tap:assert np.array_equal(S.want,res[0]['D0'])
    c.run(30);c.quiesce();c.check()
    lead={k for k,sl in c.R[0].X.slot_of.items()}
    assert lead=={(S.layers[i],int(e)) for i,e in zip(*np.nonzero(S.state==2))}
    for rt in c.R[1:]:
        r=rt.F.check();assert r is not None and r[0],('follower != leader after return',rt.rank,r,rt.F.check_msg)
        assert set(rt.X.slot_of)==lead,(rt.rank,sorted(set(rt.X.slot_of)^lead)[:6])
    # second cycle: same accounting
    c.fence();res2=c.borrow(n);_borrow_check(c,n,res2);c.fence()
    for rt,r in zip(c.R,res2):SB.give_back(rt,r)
    c.run(20);c.quiesce();c.check()
    for rt in c.R[1:]:r=rt.F.check();assert r is not None and r[0]
    return c

def test_tap_union():
    test_invalidation_union_return(seed=5,tap=True)

def test_many_seeds():
    for sd in range(6):test_invalidation_union_return(seed=10+sd,tap=bool(sd%2))

def test_cant_and_extras():
    c=Cluster(7);c.run(30);c.fence()
    free0=[len(rt.X.free) for rt in c.R]
    res=c.borrow(10,cant=(2,))
    assert res==[None]*3 and all(not rt.X.lent for rt in c.R) and [len(rt.X.free) for rt in c.R]==free0
    # a follower with an extra resident the leader does not know: its extras join U on every rank
    rt=c.R[1];X=rt.X;k=(LAY[0],NE-1)
    if k not in X.slot_of and X.free:
        sl=X.free.pop();X.slot_of[k]=sl;X._stage_apply({k[0]:[(k[1],X._row(k[0],k[1],4,sl))]})
        X.slots[sl,:8]=torch.tensor(list(int(k[0]*NE+k[1]).to_bytes(8,'little')),dtype=torch.uint8)
    S=c.R[0].S;need=NSLOT-4-len(c.R[0].X.free)
    res=c.borrow(NSLOT-4,margin=0);assert all(r is not None for r in res);_borrow_check(c,NSLOT-4,res)
    assert res[0]['U']>need,('rank 1 extras did not join U',res[0]['U'],need)
    c.R[1].X.evict_now([k]) if k in X.slot_of else None
    c.fence()
    for rt,r in zip(c.R,res):SB.give_back(rt,r)
    c.run(20);c.quiesce();c.check()

def test_lent_slots_never_freed():
    c=Cluster(9,ranks=1);c.run(20);c.fence();X=c.R[0].X;S=c.R[0].S
    res=c.borrow(10)[0];span=set(range(res['a'],res['a']+10))
    k=next(iter(X.slot_of));sl=X.slot_of[k]
    X.lent.add(sl);f0=list(X.free);X._rel(sl);assert X.free==f0;X.lent.discard(sl)
    # cancel / failed reads / normal downs while lent never free a lent slot; apply never takes one
    c.R[0].eng.fail={L*NE+E for L in LAY for E in range(NE) if E%3==0}
    for _ in range(60):
        c.run(1);c.check();assert not (set(X.free)&span) and X.lent==span
        for kk in list(X.pend)[:1]:X.cancel_up(*kk,S)
    c.fence();SB.give_back(c.R[0],res);assert span<=set(X.free)

def test_abort_returns_once():
    c=Cluster(11,ranks=1);c.run(20);rt=c.R[0]
    rt.lay={L:{'M':types.SimpleNamespace(table=c.R[0].X.layers[L][1].table),'MB':c.R[0].X.layers[L][1]} for L in LAY}
    lm=LP.LM(rt);lm.bw='all';lm.C=2;lm.per_tok=64;lm.ring_layers=LAY;lm.vidx={L:i for i,L in enumerate(LAY)};lm.mnbt=4096
    lm.ring=LP.Ring(types.SimpleNamespace(upgrade=lambda *a:None,poll=lambda:[]),RB,2,rt.dev,rt.X.rc,lambda L,E:L*NE+E,own=False)
    lm.acc={m:torch.zeros((),dtype=torch.int64) for m in ('full','bud')};lm.tot={'full':0,'bud':0};lm.sp=None
    lm.ar=lambda v:np.asarray(v,np.int32)
    stop=[False]
    def loop():                       # the streaming loop, honouring ncap / lm_hold like nq_vllm.Runtime.loop
        while not stop[0]:
            with rt.cv:
                rt.cv.wait_for(lambda:not rt.lm_hold);rt.in_iter=True;cap=rt.ncap>0
            try:c.it(rt,issue=not cap)
            finally:
                with rt.cv:rt.in_iter=False;rt.cv.notify_all()
            import time;time.sleep(1e-4)
    th=threading.Thread(target=loop,daemon=True);th.start()
    try:
        so=types.SimpleNamespace(nq_lmpf=dict(m='bud',rid='r1',start=0,n=100,end=5000,budget_s=0.),total_num_scheduled_tokens=100)
        def boom(*a,**k):raise RuntimeError('mid-window failure')
        try:lm.step(boom,so,(),{});assert False
        except RuntimeError as e:assert 'mid-window' in str(e)
        assert lm.bwi is not None and rt.X.lent and lm.bwst['borrows']==1
        assert lm.ring.base==rt.X.addr(lm.bwi['a'])
        # same rid again (op mode): borrow kept, no new one
        lm.step(lambda *a,**k:'ok',so,(),{});assert lm.bwst['borrows']==1
        r=lm.step(lambda *a,**k:'decode',types.SimpleNamespace(nq_lmpf=None,total_num_scheduled_tokens=1),(),{})
        assert r=='decode' and lm.bwi is None and not rt.X.lent and lm.bwst['returns']==1 and lm.ring.base is None
        lm.step(lambda *a,**k:'decode',types.SimpleNamespace(nq_lmpf=None,total_num_scheduled_tokens=1),(),{})
        assert lm.bwst['returns']==1
        # off file: every rank refuses, the step runs unhooked
        open(LP.BW_OFF,'w').close()
        try:assert lm.step(lambda *a,**k:'orig',so,(),{})=='orig' and lm.bwi is None and lm.bwst['cant']==1
        finally:os.remove(LP.BW_OFF)
        # rid change returns
        lm.step(lambda *a,**k:'ok',so,(),{});assert lm.bwi is not None
        so2=types.SimpleNamespace(nq_lmpf=dict(m='bud',rid='r2',start=0,n=100,end=5000,budget_s=0.,last=True),total_num_scheduled_tokens=100)
        lm.step(lambda *a,**k:'ok',so2,(),{});assert lm.bwst['returns']==3 and lm.bwi is None and not rt.X.lent   # r1 returned, r2 borrowed + returned at last
    finally:
        stop[0]=True;th.join(2)
    c.fence();c.check()

def test_size_plan_borrow():
    rb=2_621_440;pt=48*1024+512;nslot=6000;B=int(.5*nslot)
    W,C=LP.size_plan((B-2)*rb,rb,pt,65536,4096,256)
    lm=LP.LM(types.SimpleNamespace(rank=0,X=types.SimpleNamespace(rb=rb)));lm.bw='all';lm.C=C;lm.per_tok=pt
    assert W==65536 and C==256 and lm.bw_need(W)<=B
    lm.bw='ring';assert lm.bw_need(W)==2*C
    W,C=LP.size_plan((300-2)*rb,rb,pt,65536,4096,256);lm.bw='all';lm.C=C;assert lm.bw_need(W)<=300 and W>=8192
    # BORROW=0: the prod sizing is the same function, unchanged
    G=2**30;assert LP.size_plan(10*G,rb,pt,32768,4096,256)==(32768,256)

def test_plan_cols():
    rng=random.Random(0)
    for _ in range(200):
        it=[]
        for i in range(rng.randint(1,30)):
            st=rng.randint(-1,20);it.append((st,st+rng.randint(1,8),i,rng.choice([1,16,100,4096,12288])))
        cols,tot=SB.plan_cols(it)
        for a in it:
            c,w=cols[a[2]];assert c%16==0 and w%16==0 and w>=a[3] and c+w<=tot
            for b in it:
                if a[2]>=b[2] or a[1]<=b[0] or b[1]<=a[0]:continue      # boundary ranges [st, en) disjoint
                d=cols[b[2]];assert c+w<=d[0] or d[0]+d[1]<=c,('overlap',a,b,cols[a[2]],d)
        peak=max(sum(-(-x[3]//16)*16 for x in it if x[0]<=t<x[1]) for t in range(-1,30))
        assert tot>=peak

def test_arena_place():
    lm=LP.LM(types.SimpleNamespace(rank=0));nk=4;o=8
    na,nb,nc,nd,ne,nf='a','b','c','d','e','f'
    plan={na:(0,16),nb:(16,32),nc:(48,16),nd:(64,16),ne:(80,16),nf:(96,16)}
    lm.cg=dict(plan=plan,bpt={na:16,nb:32,nc:16,nd:16,ne:16,nf:16},born=[[na,nb,nc,nd,ne,nf]],PTa=112)
    lm.arena=torch.zeros(16*112,dtype=torch.uint8)
    g=torch.arange(8,dtype=torch.float32);base=torch.arange(8,dtype=torch.float32)
    va=torch.arange(16,dtype=torch.float32).view(4,4);vb=torch.arange(16,dtype=torch.float64).view(4,4)
    vc=base[:4].view(4,1).expand(4,4).contiguous() if False else torch.ones(4,4,dtype=torch.float32)
    al=torch.arange(32,dtype=torch.float32);vd=al[:16].view(4,4);ve=al[16:].view(4,4)        # alias pair: stays
    vf=g[:4].view(4,1).repeat(1,4)
    vg=g[:4].view(4,1).expand(4,4)                                                              # non-contiguous: stays
    env={na:va,nb:vb,nc:vc,nd:vd,ne:ve,nf:vf}
    lm.arena_place(0,env,o,nk,{g.untyped_storage().data_ptr()})
    for n,v in ((na,va),(nb,vb),(nc,vc),(nf,vf)):
        assert env[n].untyped_storage().data_ptr()==lm.arena.untyped_storage().data_ptr() and torch.equal(env[n],v)
        assert env[n].data_ptr()==lm.arena.data_ptr()+o*112+plan[n][0]*nk
    assert env[nd] is vd and env[ne] is ve and lm.bwst['fb_alias']==2 and lm.bwst['place']==4
    # an arena view of a placed value is copied into its own columns
    lm.cg['born']=[[nd]];lm.cg['bpt'][nd]=16;env2={nd:env[na]};lm.arena_place(0,env2,0,nk,set())
    assert lm.bwst['arena_view']==1 and torch.equal(env2[nd],va) and env2[nd].data_ptr()==lm.arena.data_ptr()+64*nk
    lm.cg['born']=[[nd]];env3={nd:vg};lm.arena_place(0,env3,0,nk,set());assert env3[nd] is vg and lm.bwst['fb_shape']==1
    env4={nd:g[:4].view(4,1).repeat(1,4)};lm.arena_place(0,env4,0,nk,{env4[nd].untyped_storage().data_ptr()});assert lm.bwst['fb_g']==1

def test_colplan_online():
    """rows arrive in order, each row's values by size: the online plan is the offline plan; a stride bounds it"""
    rng=random.Random(1)
    for _ in range(200):
        it=[];k=0
        for st in range(-1,20):
            for _ in range(rng.randint(0,3)):it.append((st,st+rng.randint(1,8),k,rng.choice([1,16,100,4096,12288])));k+=1
        if not it:continue
        cols,tot=SB.plan_cols(it)
        P=SB.ColPlan(None)
        for st in range(-1,20):
            for a in sorted([x for x in it if x[0]==st],key=lambda x:(-x[3],str(x[2]))):P.add(*a)
        assert P.cols==cols and P.tot==tot
        Q=SB.ColPlan(tot//2+16);got=0
        for a in sorted(it,key=lambda x:(x[0],-x[3],str(x[2]))):
            r=Q.add(*a)
            if r is not None:got+=1;assert r[0]+r[1]<=tot//2+16
        assert Q.tot<=tot//2+16

def test_arena_place_lazy():
    """no FX meta: columns learned from the first chunk's values (row order); later chunks reuse them; a value that
    does not fit in the stride, or has no per-token size, stays in the allocator"""
    lm=LP.LM(types.SimpleNamespace(rank=0));nk=4
    P=SB.ColPlan(64);P.add(-1,2,'__emb',16)
    xr={'a':(0,2),'b':(0,1),'c':(1,2),'d':(1,2),'s':(0,2)}
    lm.cg=dict(plan=P.cols,bpt={},born=[['a','b','s'],['c','d']],PTa=64,cplan=P,xr=xr,bad=set())
    lm.arena=torch.zeros(12*64,dtype=torch.uint8)
    for o in (0,4,8):
        env={'a':torch.full((nk,4),1.+o),'b':torch.full((nk,2),2.+o,dtype=torch.float64),'s':torch.tensor(3.)}
        lm.arena_place(0,env,o,nk,set())
        env.pop('b')                                                   # last use row 0 -> its column is free for row 1
        env.update(c=torch.full((nk,4),4.+o),d=torch.full((nk,16),5.+o))
        lm.arena_place(1,env,o,nk,set())
        A=lm.arena.untyped_storage().data_ptr()
        for n,val in (('a',1.),('c',4.)):
            c=P.cols[n][0];assert env[n].untyped_storage().data_ptr()==A and env[n].data_ptr()==lm.arena.data_ptr()+o*64+c*nk
            assert torch.all(env[n]==val+o)
        assert env['d'].untyped_storage().data_ptr()!=A and env['s'].untyped_storage().data_ptr()!=A
    assert P.cols['__emb']==(0,16) and P.cols['a']==(16,16) and P.cols['b']==(32,16) and P.cols['c']==(32,16)
    assert 'd' in lm.cg['bad'] and 's' in lm.cg['bad'] and lm.bwst['fb_plan']==1 and lm.cg['bpt']=={'a':16,'b':16,'c':16}

if __name__=='__main__':
    fs=[(n,f) for n,f in list(globals().items()) if n.startswith('test_') and callable(f)];bad=0
    for n,f in fs:
        try:f();print('ok  ',n,flush=True)
        except Exception:
            import traceback;bad+=1;print('FAIL',n);traceback.print_exc()
    print(f'{len(fs)-bad}/{len(fs)} passed');sys.exit(1 if bad else 0)
