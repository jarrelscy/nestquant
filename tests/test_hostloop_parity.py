"""nq-io upgrade 5 parity (NQ_HOSTLOOP=cpp vs the Python host loop). Needs no GPU.
  python tests/test_hostloop_parity.py follower [oplog files ...]   follower replay on the real leader op log
  python tests/test_hostloop_parity.py leader                       leader Scheduler.step on real routing counts
  python tests/test_hostloop_parity.py bench                        offline ms/iter, py vs cpp
Follower: Follower vs CppFollower(coalesce=False) and CoalescingFollower vs CppFollower(coalesce=True) drive two
identical fake executors (tests/test_follower_coalesce.FakeX: slot pool, FIFO slot wait, async completion, read
errors, best-effort cancel). Every executor call (apply(ups, downs) lists in order, cancel_up) is traced; the traces,
the landed sets, the backlogs, the stats and the live check must be identical at every tick. The C++ side reads the
log as raw int32 chunks (feed) with records split across reads and rotation markers, as the serve's OpLog.get_into.
Leader: two Schedulers (hostloop='py' / 'cpp') in lockstep on per-step routing counts from the GLM-5.3 routing logs
(/data/Jarrel/routing_logs/glm5.3-arvq-v2, one model step per (file, step id)); a fake executor feeds back
landed / released / failed (incl. read errors) identically. Per step: ups and downs identical (same order), and
score / state / want / hold / budget / stats bitwise equal. Predictors: EMA (serve config and a budget/slot-bound
config with a fixed set, kv_pressure and pin), a stub with float32 ties / NaN / float16 order keys, and the jF joint
predictor (GPUJointPredictor on CPU) recorded on the py side and replayed on the cpp side (the resident masks the
two schedulers hand it must match)."""
import os,sys,glob,random,time,collections
HERE=os.path.dirname(os.path.abspath(__file__));sys.path.insert(0,HERE+'/../streaming');sys.path.insert(0,HERE)
import numpy as np
import oplog as OL,scheduler as SC
from test_follower_coalesce import load,FakeX,leader_tables
MAGIC=OL.MAGIC
LAYERS=list(range(3,78));NE=256

# ---------------------------------------------------------------- follower
class TraceX(FakeX):
    def __init__(s,*a,**k):super().__init__(*a,**k);s.trace=[]
    def apply(s,ups,downs,sched):
        s.trace.append(('apply',tuple(map(tuple,ups)),tuple(map(tuple,downs))));super().apply([tuple(x) for x in ups],[tuple(x) for x in downs],sched)
    def cancel_up(s,L,E,sched):
        r=super().cancel_up(L,E,sched);s.trace.append(('cancel',L,E,r));return r

class PyLog:
    def __init__(s):s.buf=[]
    def get(s):r=s.buf;s.buf=[];return r
class WordLog:
    """int32 word stream like the shared-memory log: get_into hands everything written so far (maybe ending mid-record)"""
    def __init__(s):s.w=np.zeros(0,np.int32);s.rot=0
    def write(s,recs,frag=None,rot=False):
        parts=[s.w]
        for u,d in recs:parts.append(np.array([MAGIC,len(u),len(d)]+[x for p in list(u)+list(d) for x in p],np.int32))
        if rot:parts.append(np.array([MAGIC,-1,0],np.int32))
        s.w=np.concatenate(parts)
    def get_into(s,feed):
        while True:
            used,rot=feed(np.ascontiguousarray(s.w));s.w=s.w[used:]
            if not rot:return
            s.rot+=1

def enc(r):u,d=r;return np.array([MAGIC,len(u),len(d)]+[x for p in list(u)+list(d) for x in p],np.int32)

def follower_parity(recs,coal,nslot,rate,perr,seed,init):
    ck=sorted(set([1]+list(range(500,len(recs),max(1,len(recs)//25)))+[len(recs)]))
    XA=TraceX(nslot,rate,(2,12),perr,seed);XB=TraceX(nslot,rate,(2,12),perr,seed)
    LA=PyLog();LB=WordLog()
    A=(OL.CoalescingFollower if coal else OL.Follower)(XA,LA);B=OL.CppFollower(XB,LB,LAYERS,coalesce=coal)
    for F in (A,B):F.mark_busy(init);F.enable_check(init)
    for k in init:XA.apply([k],[],A);XB.apply([k],[],B)
    rng=random.Random(seed+1);i=0;tail=None;bad=0;nt=0;checks=0
    def cmp(where):
        nonlocal bad
        if XA.trace!=XB.trace:
            n=next(j for j,(a,b) in enumerate(zip(XA.trace,XB.trace)) if a!=b) if any(a!=b for a,b in zip(XA.trace,XB.trace)) else min(len(XA.trace),len(XB.trace))
            print(f'  TRACE DIFF {where} at call {n}: py {XA.trace[n:n+1]} cpp {XB.trace[n:n+1]}');bad+=1;return False
        XA.trace.clear();XB.trace.clear()
        if A.up!=B.up or A.backlog()!=B.backlog() or dict(A.stats)!=B.stats:
            print(f'  STATE DIFF {where}: up {len(A.up)}/{len(B.up)} backlog {A.backlog()}/{B.backlog()} stats {dict(A.stats)} {B.stats}');bad+=1;return False
        m=np.zeros((len(LAYERS),NE),bool)
        for L,E in A.up:m[L-3,E]=True
        if not np.array_equal(m,B.mask):print(f'  MASK DIFF {where}');bad+=1;return False
        return True
    for c in ck:
        while i<c and not bad:
            k=min(c,i+rng.randint(1,6));new=recs[i:k];i=k
            LA.buf.extend(new)
            # the cpp side: records through the word stream, sometimes the next record half written, sometimes a rotation
            words=[enc(r) for r in new]
            if tail is not None:words[0]=words[0][tail:]
            tail=None
            if i<c and rng.random()<0.3:
                nx=enc(recs[i]);tail=rng.randint(1,len(nx)-1);words.append(nx[:tail])
            if rng.random()<0.05 and tail is None:words.append(np.array([MAGIC,-1,0],np.int32))
            LB.w=np.concatenate([LB.w]+words)
            A.step();B.step();XA.tick(A);XB.tick(B);nt+=1
            if not cmp(f'tick {nt} rec {i}'):break
        if bad:break
        if tail is not None:          # finish the half-written record before draining
            nx=enc(recs[i]);LB.w=np.concatenate([LB.w,nx[tail:]]);LA.buf.append(recs[i]);i+=1;tail=None
        while (A.backlog() or XA.busy() or B.backlog() or XB.busy()) and not bad:
            A.step();B.step();XA.tick(A);XB.tick(B);nt+=1
            if not cmp(f'drain tick {nt}'):break
        if bad:break
        ra,rb_=A.check(),B.check();checks+=1
        if ra!=rb_:print(f'  CHECK DIFF rec {i}: py {ra} cpp {rb_}');bad+=1;break
    return bad==0,dict(ticks=nt,checks=checks,reads=XA.reads,cancels=XA.cancels,errs=XA.errs,rot=LB.rot,stats=B.stats,check=ra)

def prep_recs(files):
    recs=load(files)
    pre=set();seen=set()
    for ups,downs in recs:
        for k in downs:
            if k not in seen:pre.add(k)
            seen.add(k)
        for k in ups:seen.add(k)
    return recs,sorted(pre)

def main_follower(files):
    files=files or sorted(glob.glob('/data/Jarrel/nq-io/oplogs/nq_oplog_*'),key=lambda f:int(f.rsplit('.',1)[1]))
    recs,init=prep_recs(files);NREC=int(os.environ.get('NREC',len(recs)));recs=recs[:NREC]
    print(f'follower parity: {len(recs)} records, init {len(init)} experts')
    ok=True
    for coal in (False,True):
        for rate,perr in ((40,0.0),(40,0.002),(8,0.002)):
            t=time.time();r,info=follower_parity(recs,coal,6000,rate,perr,7,init);ok&=r
            print(f'  coalesce={coal!s:5s} rate {rate:3d} perr {perr}: {"IDENTICAL" if r else "DIFFER"}  ticks {info["ticks"]} checkpoints {info["checks"]} '
                  f'reads {info["reads"]} cancels {info["cancels"]} read_err {info["errs"]} rotations {info["rot"]} final check {info["check"]} ({time.time()-t:.0f}s)')
    return ok

# ---------------------------------------------------------------- leader
def routing_steps(nfiles,start=0):
    """-> list of (counts [75, 256] f64, ntok, token_ids, new_request) per model step"""
    fs=sorted(glob.glob('/data/Jarrel/routing_logs/glm5.3-arvq-v2/seg-*.npz'))[start:]
    out=[];last_req=None;run=min(20,nfiles)        # runs of 20 consecutive files at evenly spaced offsets over the logs
    fs=[f for o in np.linspace(0,len(fs)-run,max(1,nfiles//run)).astype(int) for f in fs[o:o+run]]
    for f in fs:
        z=np.load(f);ex=z['experts'].astype(np.int64);st=z['step'];tid=z['token_ids'];rq=z['req']
        for sv in np.unique(st):
            m=st==sv;e=ex[m][:,3:78,:];n=int(m.sum())
            c=np.bincount((e+(np.arange(75)*NE)[None,:,None]).ravel(),minlength=75*NE).reshape(75,NE).astype(np.uint16)   # f64 at use (memory)
            r=set(rq[m].tolist());nr=last_req is not None and not r<=last_req;last_req=r
            out.append((c,n,tid[m].tolist(),nr))
    if os.environ.get('DECODE','0')=='1':out=[x for x in out if x[1]<=16]   # decode steps only (the serve's steady state)
    return out

class LeaderX:
    """fake executor: each step lands / fails a random subset of in-flight ups (some read errors), releases downs"""
    def __init__(s,seed):s.rng=random.Random(seed);s.upq=[];s.dq=[]
    def apply(s,ups,downs):s.upq+=ups;s.dq+=downs
    def tick(s,S):
        r=s.rng;keep=[]
        for k in s.upq:
            x=r.random()
            if x<0.55:S.landed(*k)
            elif x<0.57:S.failed(*k,read_error=True)
            elif x<0.58:S.failed(*k)
            else:keep.append(k)
        s.upq=keep;keep=[]
        for k in s.dq:
            if r.random()<0.7:S.released(*k)
            else:keep.append(k)
        s.dq=keep

class StubPred:
    """predictor with awkward keys: float32 ties, NaN, float16 order scores, target None at times"""
    bs=None   # wants_sal
    def __init__(s,seed,dt=np.float32):s.rng=np.random.default_rng(seed);s.n=0;s.dt=dt
    def step(s,c,ntok,tid,nr,sal=None):s.n+=1;return s.n%3==0
    def target(s,res):
        if s.n%9==0:return None
        return s.rng.random((75,NE))<0.3
    def order_score(s,res):
        if s.n%5==0:return None
        v=np.round(s.rng.random((75,NE))*8).astype(s.dt);v[s.rng.random((75,NE))<0.05]=np.nan
        v[res]+=1;return v
    def close(s):pass

class RecPred:
    """py side: the real predictor, recording each call; play(): the cpp side replays and asserts identical inputs"""
    def __init__(s,P):s.P=P;s.log=collections.deque();s.bs=getattr(P,'bs',None);s.wants=hasattr(P,'bs')
    def step(s,*a,**k):r=s.P.step(*a,**k);s.log.append(('step',r));return r
    def target(s,res):r=s.P.target(res);s.log.append(('target',res.copy(),r));return r
    def order_score(s,res):r=s.P.order_score(res);s.log.append(('order',res.copy(),r));return r
    def close(s):pass
class PlayPred:
    def __init__(s,R):s.R=R;s.bs=R.bs;s.miss=0
    def step(s,*a,**k):t,r=s.R.log.popleft();assert t=='step';return r
    def target(s,res):
        t,want,r=s.R.log.popleft();assert t=='target'
        if not np.array_equal(want,res):s.miss+=1
        return None if r is None else r.copy()
    def order_score(s,res):
        t,want,r=s.R.log.popleft();assert t=='order'
        if not np.array_equal(want,res):s.miss+=1
        return None if r is None else r.copy()
    def close(s):pass

def eq(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()

def leader_parity(name,steps,mk,cfg,seed=0,kv=False,pin=False):
    """mk(): -> (pred_py, pred_cpp)"""
    pa,pb=mk()
    rb=2560000
    A=SC.Scheduler(LAYERS,cfg['fixed'],cfg['dflt'],rb,NE=NE,n_float=cfg['nf'],slots=cfg['slots'],cap_GBps=cfg['cap'],predictor=pa,hostloop='py')
    B=SC.Scheduler(LAYERS,cfg['fixed'],cfg['dflt'],rb,NE=NE,n_float=cfg['nf'],slots=cfg['slots'],cap_GBps=cfg['cap'],predictor=pb,hostloop='cpp')
    assert A.core is None and B.core is not None
    XA,XB=LeaderX(seed),LeaderX(seed)
    init=[(L,E) for L in LAYERS for E in cfg['dflt'][L] if E not in cfg['fixed'][L]][:cfg['slots']]
    for S,X in ((A,XA),(B,XB)):
        for L,E in init:S.state[S.li[L],E]=1
        X.apply(init,[])
    rng=random.Random(seed+5);nops=0;tA=tB=0.0
    for n,(c,ntok,tid,nr) in enumerate(steps):
        c=c.astype(np.float64)
        sal=c*1.7 if A.wants_sal else None
        t=time.perf_counter();ua,da=A.step(c,ntok,tid,nr,sal=sal);t2=time.perf_counter();ub,db=B.step(c,ntok,tid,nr,sal=sal);t3=time.perf_counter()
        tA+=t2-t;tB+=t3-t2
        if (ua,da)!=(ub,db):
            print(f'  {name}: OPS DIFF at step {n}: ups {len(ua)}/{len(ub)} downs {len(da)}/{len(db)} first ups {ua[:3]} {ub[:3]}');return False
        for f in ('score','state','want','hold'):
            if not eq(getattr(A,f),getattr(B,f)):print(f'  {name}: {f} DIFF at step {n}');return False
        if A.budget!=B.budget or A.stats!=B.stats or A.tok!=B.tok or A.next_refresh!=B.next_refresh:
            print(f'  {name}: scalar DIFF at step {n}: budget {A.budget} {B.budget} stats {A.stats} {B.stats}');return False
        if isinstance(pb,PlayPred) and pb.miss:print(f'  {name}: predictor input (resident mask) DIFF at step {n}');return False
        XA.apply(ua,da);XB.apply(ub,db);XA.tick(A);XB.tick(B);nops+=len(ua)+len(da)
        if kv and rng.random()<0.01:
            k=rng.randint(1,40)
            if A.kv_pressure(k)!=B.kv_pressure(k):print(f'  {name}: kv_pressure DIFF at step {n}');return False
        if pin and n==len(steps)//3:
            pm=np.random.default_rng(seed).random((75,NE))<0.05;A.pin=pm.copy();B.pin=pm.copy()
        if pin and n==2*len(steps)//3:A.pin=B.pin=None
    print(f'  {name:28s} IDENTICAL over {len(steps)} steps, {nops} ops, stats {A.stats}; step() py {1e3*tA/len(steps):.3f} cpp {1e3*tB/len(steps):.3f} ms/iter (incl. predictor)')
    return True

def cfgs():
    rng=np.random.default_rng(3)
    k0=dict(fixed={L:[] for L in LAYERS},nf=77,slots=6000,cap=1e6)
    k0['dflt']={L:[int(x) for x in rng.permutation(NE)[:77]] for L in LAYERS}
    fx={L:sorted(int(x) for x in rng.permutation(NE)[:26]) for L in LAYERS}
    b=dict(fixed=fx,nf=51,slots=3000,cap=6.0)
    b['dflt']={L:[int(x) for x in rng.permutation(NE) if x not in fx[L]][:60] for L in LAYERS}
    return k0,b

def jf_pair():
    sys.path.insert(0,'/data/Jarrel/nq-serve/predictor/joint')
    import gpu_predictor as GJ,torch
    torch.set_num_threads(int(os.environ.get('NQ_TEST_THREADS','8')))
    P=GJ.GPUJointPredictor(LAYERS,{L:[] for L in LAYERS},'/data/Jarrel/nq-serve/predictor/joint/jF.pt',n_float=77,hm=0.7,device='cpu',
                           v2_model='/data/Jarrel/nq-serve/predictor/joint/v2_sal_tweedie1.5.txt')
    R=RecPred(P);return R,PlayPred(R)

def main_leader():
    nf=int(os.environ.get('NFILES','200'))
    steps=routing_steps(nf);print(f'leader parity: {len(steps)} model steps, {sum(s[1] for s in steps)} tokens from {nf} routing-log files')
    k0,b=cfgs();ok=True
    ok&=leader_parity('ema / serve k0 cfg',steps,lambda:('ema','ema'),k0)
    ok&=leader_parity('ema / fixed+budget+slots',steps,lambda:('ema','ema'),b,seed=1,kv=True,pin=True)
    ok&=leader_parity('stub f32 ties+NaN',steps,lambda:(StubPred(4),StubPred(4)),b,seed=2,kv=True)
    ok&=leader_parity('stub f16 (numpy fallback)',steps,lambda:(StubPred(5,np.float16),StubPred(5,np.float16)),k0,seed=3)
    ok&=leader_parity('stub f64',steps,lambda:(StubPred(6,np.float64),StubPred(6,np.float64)),b,seed=4)
    if os.environ.get('NQ_TEST_JF','1')=='1':
        js=[x for x in steps if x[1]<=16][:int(os.environ.get('JF_STEPS','1500'))]   # decode steps (jF scores ntok<=16 steps only)
        ok&=leader_parity('jF joint (CPU) rec/replay',js,jf_pair,k0,seed=5,kv=True)
    return ok

# ---------------------------------------------------------------- bench
def main_bench():
    """ms per host-loop iteration. Leader: Scheduler.step with the EMA rule and with a replayed predictor (the predictor's
    own cost excluded), serve k0 config. Follower: one step() per 4 ms poll at the live record rate, and the drain of a
    deep backlog (the follower cost that grows with backlog)."""
    steps=[x for x in routing_steps(int(os.environ.get('NFILES','150'))) if x[1]<=16];k0,_=cfgs()
    for name,mkp in (('ema',lambda:'ema'),('stub-f32',lambda:StubPred(4))):
        for hl in ('py','cpp'):
            S=SC.Scheduler(LAYERS,k0['fixed'],k0['dflt'],2560000,NE=NE,n_float=77,slots=6000,cap_GBps=1e6,predictor=mkp(),hostloop=hl);X=LeaderX(0)
            ts=[]
            for c,ntok,tid,nr in steps:
                c=c.astype(np.float64);t=time.perf_counter();u,d=S.step(c,ntok,tid,nr);ts.append(time.perf_counter()-t);X.apply(u,d);X.tick(S)
            ts=np.array(ts)*1e3;print(f'  leader {name:8s} {hl}: mean {ts.mean():.3f} p50 {np.median(ts):.3f} p99 {np.percentile(ts,99):.3f} ms/iter')
    files=sorted(glob.glob('/data/Jarrel/nq-io/oplogs/nq_oplog_*'),key=lambda f:int(f.rsplit('.',1)[1]))
    recs,init=prep_recs(files);recs=recs[:int(os.environ.get('NREC','8000'))]
    for coal in (False,True):
        for hl in ('py','cpp'):
            for rate in (40,6):           # 40 reads/tick keeps up (live-like); 6 = a drive that falls behind (backlog grows)
                X=FakeX(6000,rate,(2,12),0.0,7);L=PyLog() if hl=='py' else WordLog()
                F=(OL.CoalescingFollower if coal else OL.Follower)(X,L) if hl=='py' else OL.CppFollower(X,L,LAYERS,coalesce=coal)
                F.mark_busy(init)
                for k in init:X.apply([k],[],F)
                ts=[];bl=[];i=0;rng=random.Random(1)
                while i<len(recs):
                    k=min(len(recs),i+rng.randint(1,3))
                    if hl=='py':L.buf.extend(recs[i:k])
                    else:L.write(recs[i:k])
                    i=k;t=time.perf_counter();F.step();ts.append(time.perf_counter()-t);X.tick(F);bl.append(F.backlog())
                ts=np.array(ts)*1e3
                print(f'  follower coalesce={coal!s:5s} {hl:3s} rate {rate:2d}: backlog mean {np.mean(bl):8.0f} max {max(bl):7d}: step mean {ts.mean():.3f} p50 {np.median(ts):.3f} p99 {np.percentile(ts,99):.3f} ms ({len(ts)} steps)',flush=True)
    return True

if __name__=='__main__':
    mode=sys.argv[1] if len(sys.argv)>1 else 'all'
    ok=True
    if mode in ('follower','all'):ok&=main_follower(sys.argv[2:])
    if mode in ('leader','all'):ok&=main_leader()
    if mode=='bench':main_bench()
    if mode!='bench':print('PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 1)
