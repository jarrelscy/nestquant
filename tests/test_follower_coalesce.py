"""Follower replay correctness (nq-io): replay a real leader op log through the in-order Follower and the
CoalescingFollower against a fake executor with RankExecutor's semantics (slot pool + FIFO slot wait, one outstanding
op per expert, asynchronous completion at a modelled drive rate, read errors, best-effort cancel of ops not yet
started), and check after every checkpoint (log consumed up to record n, then run until quiescent):
    follower level-4 set == leader level-4 set after record n   (minus experts whose read failed)
for both followers (and the serve's live check, Follower.check(), must agree), plus slot-pool consistency at every tick (in use == landed + in flight + draining <= nslot).
  python tests/test_follower_coalesce.py [oplog files ...]   (default: the jF serve log in /data/Jarrel/nq-io/oplogs)
Prints SSD reads issued / cancelled / backlog for both followers (the I/O the coalescer saves)."""
import os,sys,glob,random,collections
HERE=os.path.dirname(os.path.abspath(__file__));sys.path.insert(0,HERE+'/../streaming')
import numpy as np,oplog as OL
MAGIC=OL.MAGIC

def load(files):
    recs=[]
    for f in files:
        a=np.fromfile(f,np.int32);i=0
        while i+3<=len(a):
            assert a[i]==MAGIC
            nu,nd=int(a[i+1]),int(a[i+2])
            if nu<0:break
            j=i+3+2*(nu+nd)
            if j>len(a):break
            p=a[i+3:j].reshape(-1,2).tolist();recs.append(([tuple(x) for x in p[:nu]],[tuple(x) for x in p[nu:]]));i=j
    return recs

class FakeLog:
    def __init__(s):s.buf=[]
    def get(s):r=s.buf;s.buf=[];return r

class FakeX:
    """RankExecutor stand-in. rate = reads started per tick (drive throughput), lat = ticks per read, perr = read-error
    probability (deterministic per (L, E, attempt) so both followers see the same drive)."""
    def __init__(s,nslot,rate=8,lat=(2,12),perr=0.0,seed=0):
        s.nslot=nslot;s.free=nslot;s.pend=[];s.eq=collections.deque();s.fl={};s.t=0;s.rate=rate;s.lat=lat;s.perr=perr
        s.rng=random.Random(seed);s.out={};s.reads=0;s.cancels=0;s.errs=0;s.slot_of=set();s.done=[];s.draining=set()
    def apply(s,ups,downs,sched):
        for k in downs:
            assert k not in s.out,('two outstanding ops',k);s.out[k]='down';s.done.append((s.t+1,k,'down'))
        for k in ups:
            assert k not in s.out,('two outstanding ops',k)
            if s.free==0:s.pend.append(k);s.out[k]='wait';continue
            s.free-=1;s.slot_of.add(k);s.out[k]='queued';s.eq.append(k)
    def cancel_up(s,L,E,sched):
        k=(L,E)
        if s.out.get(k)=='wait':s.pend.remove(k);del s.out[k];s.cancels+=1;sched.cancelled(L,E);return True
        if s.out.get(k)=='queued':s.out[k]='cancel';return True
        return False
    def tick(s,sched):
        s.t+=1;n=0
        while s.eq and n<s.rate:
            k=s.eq.popleft()
            if s.out[k]=='cancel':
                del s.out[k];s.free+=1;s.slot_of.discard(k);s.cancels+=1;sched.cancelled(*k);continue
            s.out[k]='reading';s.reads+=1;n+=1;s.done.append((s.t+s.rng.randint(*s.lat),k,'up'))
        keep=[]
        for td,k,kind in s.done:
            if td>s.t:keep.append((td,k,kind));continue
            del s.out[k]
            if kind=='up':
                if s.perr and random.Random(hash((k,s.reads))).random()<s.perr:
                    s.errs+=1;s.free+=1;s.slot_of.discard(k);sched.failed(*k,read_error=True)
                else:sched.landed(*k)
            else:s.free+=1;s.slot_of.discard(k);sched.released(*k)
        s.done=keep
        while s.pend and s.free:
            k=s.pend.pop(0);del s.out[k];s.apply([k],[],sched)
        inuse=s.nslot-s.free;assert inuse==len(s.slot_of)<=s.nslot,(inuse,len(s.slot_of))
    def busy(s):return bool(s.out)

def leader_tables(recs,ck):
    cur=set();snap={}
    for n,(ups,downs) in enumerate(recs,1):
        for k in downs:assert k in cur,('leader downs an expert it does not hold',k,n);cur.discard(k)
        for k in ups:assert k not in cur,('leader ups an expert it holds',k,n);cur.add(k)
        if n in ck:snap[n]=set(cur)
    return snap

def run(recs,cls,ck,snap,nslot,rate,lat,perr,seed,init=()):
    log=FakeLog();X=FakeX(nslot,rate,lat,perr,seed);F=cls(X,log);F.enable_check(());rng=random.Random(seed+1)
    i=0;maxb=0;bl=[];failed=set();fails=0
    F_failed=F.failed
    def failed_hook(L,E,read_error=False):failed.add((L,E));F_failed(L,E,read_error)
    F.failed=failed_hook
    for c in sorted(ck):
        while i<c:
            k=min(c,i+rng.randint(1,6));log.buf.extend(recs[i:k]);i=k
            F.step();X.tick(F);b=len(F.q);maxb=max(maxb,b);bl.append(b)
        while F.q or X.busy():F.step();X.tick(F)
        want=snap[c];got=F.up
        # an expert whose read failed stays at level 2 here until the log ups it again (then it is retried)
        bad=(want^got)-failed
        if bad:fails+=1;print(f'  {cls.__name__}: MISMATCH at record {c}: {len(bad)} experts differ, e.g. {sorted(bad)[:5]}')
        lc=F.check()          # the serve's live check (NQ_FOLLOW_CHECK) must agree with this independent one
        if lc is None or lc[0]!=(not bad):fails+=1;print(f'  {cls.__name__}: live check {lc} disagrees at record {c} (bad {len(bad)})')
        failed&=want   # keep only failures still relevant
    return dict(reads=X.reads,cancels=X.cancels,errs=X.errs,max_backlog=maxb,mean_backlog=float(np.mean(bl)),fails=fails,ticks=X.t,
                stats=dict(F.stats))

if __name__=='__main__':
    files=sys.argv[1:] or sorted(glob.glob('/data/Jarrel/nq-io/oplogs/nq_oplog_*'),key=lambda f:int(f.rsplit('.',1)[1]))
    recs=load(files);N=len(recs)
    NREC=int(os.environ.get('NREC',N));recs=recs[:NREC];N=len(recs)
    print(f'{len(files)} log files, {N} records, {sum(len(u) for u,_ in recs)} ups, {sum(len(d) for _,d in recs)} downs')
    # the log starts mid-boot: leader state before record 0 is unknown -> start from the experts it downs before upping
    pre=set();seen=set()
    for ups,downs in recs:
        for k in downs:
            if k not in seen:pre.add(k)
            seen.add(k)
        for k in ups:seen.add(k)
    recs=[(sorted(pre),[])]+recs;N=len(recs)
    nslot=int(os.environ.get('NSLOT',80*75))
    ck=sorted(set([1]+list(range(500,N,max(1,N//25)))+[N]))
    snap=leader_tables(recs,set(ck))
    print(f'leader start set {len(pre)} experts; peak leader set {max(len(v) for v in snap.values())}; nslot {nslot}; {len(ck)} checkpoints')
    ok=True
    for rate,perr in ((int(os.environ.get('RATE','40')),0.0),(int(os.environ.get('RATE','40')),0.002)):
        res={}
        for cls in (OL.Follower,OL.CoalescingFollower):
            r=run(recs,cls,ck,snap,nslot,rate,(2,12),perr,7);res[cls.__name__]=r;ok&=r['fails']==0
            print(f'rate {rate}/tick perr {perr}: {cls.__name__:20s} reads {r["reads"]:8d} cancels {r["cancels"]:6d} read_err {r["errs"]:4d} '
                  f'backlog max {r["max_backlog"]:6d} mean {r["mean_backlog"]:8.1f} ticks {r["ticks"]} checkpoint mismatches {r["fails"]}')
        a,b=res['Follower'],res['CoalescingFollower']
        print(f'  coalescer: {1-b["reads"]/a["reads"]:.1%} fewer SSD reads, {a["ticks"]/b["ticks"]:.2f}x faster drain; stats {b["stats"]}')
    print('PASS' if ok else 'FAIL');sys.exit(0 if ok else 1)
