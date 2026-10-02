"""Leader/follower level ops across the TP ranks of one host. Rank 0 runs the scheduler and appends every step's ops
to a shared log (/dev/shm); the other ranks replay the log in order, so every rank serves the same level set (a TP step
waits for its slowest rank, and independently scheduled ranks drift apart by up to ~1/3 of the floating set).
  OpLog(path, writer)   writer: put(ups, downs); reader: get() -> [(ups, downs)] complete records appended since last get
  Follower(X, log)      step(issue) reads new records and issues them through executor X, keeping one outstanding op per
                        expert (an op on an expert whose previous op is still in flight here waits, in order); a
                        downgrade of an expert that never landed here (read error) is dropped
Record: int32 [MAGIC, n_ups, n_downs, L0, E0, L1, E1, ...] (ups then downs), one os.write per record. The log rotates
every ROT bytes: [MAGIC, -1, 0] = continue in path.<gen+1>; the leader deletes generation gen-2 when it rotates."""
import os,collections,numpy as np
MAGIC=0x4E514F50;ROT=16<<20

class OpLog:
    def __init__(s,path,writer):
        s.base=path;s.writer=writer;s.gen=0;s.off=0;s.fd=None
        if writer:s._open_w()
    def _p(s,g):return f'{s.base}.{g}'
    def _open_w(s):s.fd=os.open(s._p(s.gen),os.O_WRONLY|os.O_CREAT|os.O_TRUNC|os.O_APPEND,0o600);s.off=0
    def _w(s,a):
        n=os.write(s.fd,a.tobytes());assert n==a.nbytes,(n,a.nbytes);s.off+=n
    def put(s,ups,downs):
        if not ups and not downs:return
        s._w(np.array([MAGIC,len(ups),len(downs)]+[x for p in list(ups)+list(downs) for x in p],np.int32))
        if s.off>=ROT:
            s._w(np.array([MAGIC,-1,0],np.int32));os.close(s.fd);s.gen+=1;s._open_w()
            try:os.unlink(s._p(s.gen-2))
            except FileNotFoundError:pass
    def get(s):
        out=[]
        while True:
            if s.fd is None:
                if not os.path.exists(s._p(s.gen)):return out
                s.fd=os.open(s._p(s.gen),os.O_RDONLY)
            b=os.pread(s.fd,1<<22,s.off);a=np.frombuffer(b[:len(b)//4*4],np.int32);i=0;nxt=False
            while i+3<=len(a):
                assert a[i]==MAGIC,f'oplog {s._p(s.gen)}: bad record at byte {s.off+4*i}'
                nu,nd=int(a[i+1]),int(a[i+2])
                if nu<0:nxt=True;i+=3;break
                j=i+3+2*(nu+nd)
                if j>len(a):break                             # record still being written
                p=a[i+3:j].reshape(-1,2).tolist();out.append(([tuple(x) for x in p[:nu]],[tuple(x) for x in p[nu:]]));i=j
            s.off+=4*i
            if not nxt:return out
            os.close(s.fd);s.fd=None;s.gen+=1;s.off=0
    def close(s):
        if s.fd is not None:os.close(s.fd);s.fd=None

class Follower:
    """stand-in scheduler for the executor callbacks + in-order replay of the leader's ops"""
    def __init__(s,X,log):
        s.X=X;s.log=log;s.q=collections.deque();s.busy=set();s.up=set();s.stats=dict(ups=0,downs=0,dropped=0,read_errors=0)
        s.mask=None;s.li=None       # nq-io: optional [nL, NE] bool level-4 mask kept with s.up (per-rank served share), li = {L: row}
        s.chk=None                  # nq-io check (enable_check): the leader's intended level-4 set, folded from the raw log
    def enable_check(s,init):
        """live consistency check: s.chk = the level-4 set the leader asked for up to the last consumed record (start =
        init, the floating_default every rank loads itself). Whenever this rank is quiescent (nothing pending or in
        flight) its landed set must equal s.chk minus experts whose read failed here (s.rerr, until the log touches
        them again). Mismatches are counted (stats check_bad) and the first few logged by check(); never raises."""
        s.chk=set(init);s.rerr=set();s.stats.update(check_ok=0,check_bad=0);s.check_msg=None
    def _fold(s,ups,downs):
        if s.chk is None:return
        for k in downs:s.chk.discard(k);s.rerr.discard(k)
        for k in ups:s.chk.add(k);s.rerr.discard(k)
    def check(s):
        """-> None (not quiescent / check off) or (ok, n_missing, n_extra)"""
        if s.chk is None or s.q or s.busy:return None
        want=s.chk-s.rerr;miss=len(want-s.up);extra=len(s.up-want);ok=not miss and not extra
        s.stats['check_ok' if ok else 'check_bad']+=1
        if not ok and s.check_msg is None:s.check_msg=f'missing {sorted(want-s.up)[:4]} extra {sorted(s.up-want)[:4]}'
        return ok,miss,extra
    # executor feedback
    def landed(s,L,E):
        s.busy.discard((L,E));s.up.add((L,E))
        if s.mask is not None:s.mask[s.li[L],E]=True
        if s.chk is not None:s.rerr.discard((L,E))     # a retry (the log asked again after the failure) landed
    def released(s,L,E):
        s.busy.discard((L,E));s.up.discard((L,E))
        if s.mask is not None:s.mask[s.li[L],E]=False
    def failed(s,L,E,read_error=False):
        s.busy.discard((L,E))
        if read_error:
            s.stats['read_errors']+=1
            if s.chk is not None:s.rerr.add((L,E))
    def step(s,issue=True):
        for ups,downs in s.log.get():
            s._fold(ups,downs)
            for k in downs:s.q.append((k,2))
            for k in ups:s.q.append((k,4))
        if not issue or not s.q:return
        ups=[];downs=[];keep=collections.deque();held=set()
        for k,lv in s.q:
            if k in held or k in s.busy:keep.append((k,lv));held.add(k);continue
            held.add(k)
            if lv==2:
                if k not in s.up:s.stats['dropped']+=1;continue
                downs.append(k)
            else:ups.append(k)
            s.busy.add(k)
        s.q=keep;s.stats['ups']+=len(ups);s.stats['downs']+=len(downs)
        if ups or downs:s.X.apply(ups,downs,s)
    def level_count(s):return len(s.up)

class CoalescingFollower(Follower):
    """nq-io (NQ_FOLLOW_COALESCE=1): replay the log by net effect instead of op by op.
    The log is folded per expert into the level the leader last asked for (an up+down pair, or a down+up pair,
    that is still queued collapses to nothing; up, down, up = one up). When an expert is free here, its pending level
    is compared with its level here: equal -> moot (no op), else one op. A stale upgrade (the log has since downed the
    expert) that has not started yet is cancelled in the executor (cancel_up), so it never reads the SSD.
    Invariant (tests/test_follower_coalesce.py): once quiescent after consuming the log up to record n, the level-4
    set here equals the leader's at record n (minus read errors), exactly as for the in-order Follower.
    Order: pending experts are served oldest-decision first (a re-decided expert moves to the back)."""
    def __init__(s,X,log):
        super().__init__(X,log);s.pend=collections.OrderedDict()   # (L, E) -> last level the log asked for
        s._creq=set()                                                 # stale ups we asked the executor to cancel
        s.stats.update(log_ops=0,merged=0,moot=0,cancel_req=0,cancelled=0)
        s.q=s.pend                                                    # len(F.q) = backlog, as before
    def cancelled(s,L,E):s.busy.discard((L,E));s.stats['cancelled']+=1
    def step(s,issue=True):
        for ups,downs in s.log.get():
            s._fold(ups,downs)
            for k in downs:s._add(k,2)
            for k in ups:s._add(k,4)
        if not issue or not s.pend:return
        ups=[];downs=[]
        for k,lv in list(s.pend.items()):
            if k in s.busy:
                # in flight here: a pending down means the upgrade in flight is stale -> try to cancel it once
                if lv==2 and k not in s.up and k not in s._creq:
                    s._creq.add(k)
                    if s.X.cancel_up(*k,s):s.stats['cancel_req']+=1
                continue
            del s.pend[k];s._creq.discard(k)
            if (4 if k in s.up else 2)==lv:s.stats['moot']+=1;continue
            (downs if lv==2 else ups).append(k);s.busy.add(k)
        s.stats['ups']+=len(ups);s.stats['downs']+=len(downs)
        if ups or downs:s.X.apply(ups,downs,s)
    def _add(s,k,lv):
        s.stats['log_ops']+=1
        if k in s.pend:s.stats['merged']+=1;del s.pend[k]
        s.pend[k]=lv
    def level_count(s):return len(s.up)
