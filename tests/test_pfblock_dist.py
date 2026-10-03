"""nq_pfblock over NQ_OPLOG=dist (CPU, one process, TCP on localhost): a leader and a follower PFBlock share no files.
Prefill: the follower's _ready(L, chunk) is False until it has read the leader's plan marker and every op sent before it.
Decode: _dec_cond on the follower returns once the leader published step n and the follower's reader got there; the
leader's own _dec_cond waits for 2 streaming iterations.  python tests/test_pfblock_dist.py"""
import os,sys,time,threading,types
HERE=os.path.dirname(os.path.abspath(__file__));sys.path[:0]=[HERE+'/../streaming',HERE+'/../sm120/serve']
import oplog_net as ON,nq_pfblock as PB

def rt(rank,log):
    return types.SimpleNamespace(rank=rank,iokey='dist',log=log,F=None if rank==0 else object(),X=types.SimpleNamespace(ops={}),
                                 cv=threading.Condition(),wake=threading.Event(),in_iter=False,hl_n=0,LA=None)
def stream(r,stop):            # stand-in for Runtime.loop: one iteration = (follower) read the log, then notify
    while not stop.is_set():
        r.wake.wait(0.004);r.wake.clear()
        with r.cv:r.in_iter=True
        try:
            if r.rank:r.got+=r.log.get()
        finally:
            with r.cv:r.in_iter=False;r.hl_n+=1;r.cv.notify_all()

if __name__=='__main__':
    port=29611+os.getpid()%1000;L_=list(range(3,78));box={}
    th=threading.Thread(target=lambda:box.update(L=ON.NetOpLog(('127.0.0.1',port),writer=True,rank=0,nfollow=1,wait_s=30)));th.start()
    time.sleep(0.2);F=ON.NetOpLog(('127.0.0.1',port),writer=False,rank=1,nfollow=1,wait_s=30);th.join();L=box['L']
    rl,rf=rt(0,L),rt(1,F);rf.got=[];pl,pf=PB.PFBlock(rl,L_),PB.PFBlock(rf,L_);ok=[]
    # prefill: ops for layer 10, then the plan marker of (L10, chunk 1)
    L.put([(10,5)],[]);pl.publish(10,1,L);time.sleep(0.05)
    ok.append(('follower not ready before reading the marker',not pf._ready(10,1)))
    rf.got+=F.get()
    ok.append(('follower got the ops sent before the marker',rf.got==[([(10,5)],[])]))
    ok.append(('follower ready after reading the marker',pf._ready(10,1)))
    ok.append(('chunk 2 not ready yet',not pf._ready(10,2)))
    rf.X.ops={1:(10,5,4)}
    ok.append(('not ready while a level-4 read of the layer is in flight',not pf._ready(10,1)))
    rf.X.ops={}
    # decode: both ranks wait for step 1 with streaming threads running
    stop=threading.Event();ts=[threading.Thread(target=stream,args=(r,stop),daemon=True) for r in (rl,rf)];[t.start() for t in ts]
    res={}
    def dc(p,k):res[k]=p._dec_cond(1,5.0)
    a=threading.Thread(target=dc,args=(pf,'f'));a.start();time.sleep(0.05)
    ok.append(('follower waits before the leader published step 1','f' not in res))
    L.put([(20,7)],[]);dc(pl,'l');a.join(5)
    ok.append(('leader dec_cond ok',res['l'][0]))
    ok.append(('follower dec_cond ok after the leader marker',res.get('f',(False,))[0]))
    ok.append(('follower replayed the op sent before the step marker',([(20,7)],[]) in rf.got))
    stop.set();[t.join(2) for t in ts];L.close();F.close()
    for m,v in ok:print(f'{"ok  " if v else "FAIL"} {m}')
    sys.exit(0 if all(v for _,v in ok) else 1)
