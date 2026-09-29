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
    # executor feedback
    def landed(s,L,E):s.busy.discard((L,E));s.up.add((L,E))
    def released(s,L,E):s.busy.discard((L,E));s.up.discard((L,E))
    def failed(s,L,E,read_error=False):
        s.busy.discard((L,E))
        if read_error:s.stats['read_errors']+=1
    def step(s,issue=True):
        for ups,downs in s.log.get():
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
