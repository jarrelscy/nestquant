"""NQ_OPLOG=dist transport (streaming/oplog_net.py) == the /dev/shm OpLog: a leader process writes random records
(ups / downs, borrowed-pool keys, reclaim markers, large records that span TCP segments) to both a file OpLog and a
NetOpLog; a follower process (separate process, TCP on localhost) reads the stream with random poll timing and gate
states and must get exactly the records the file reader gets under the same gates. Also checks io stats publish ->
peer_stats on the leader, and the nq_pfblock in-band markers (put_mark): whenever the follower holds marker v, it has
already received every record sent before v, and its stream position is past the marker's.  python tests/test_oplog_net.py"""
import os,sys,time,random,tempfile,multiprocessing as mp
HERE=os.path.dirname(os.path.abspath(__file__));sys.path.insert(0,HERE+'/../streaming')
import oplog as OL,oplog_net as ON

def recs(seed,n=3000):
    r=random.Random(seed);out=[]
    for i in range(n):
        if r.random()<0.01:out.append(('reclaim',r.randrange(1,4)));continue
        nu=r.choice([0,1,2,5,40,3000]) if r.random()<0.05 else r.randrange(0,6);nd=r.randrange(0,6)
        ups=[(r.randrange(3,78)+(OL.XB*r.randrange(1,3) if r.random()<0.05 else 0),r.randrange(256)) for _ in range(nu)]
        out.append(('ops',ups,[(r.randrange(3,78),r.randrange(256)) for _ in range(nd)]))
    return out

class Gate:
    def __init__(s):s.x_have=9;s.x_done=9      # every borrowed epoch / reclaim allowed (gated reads checked below)

def leader(port,path,q):
    L=ON.NetOpLog(('127.0.0.1',port),writer=True,rank=0,nfollow=1,wait_s=60);F=OL.OpLog(path,writer=True)
    r=random.Random(7);nrec=0;mv=0;before={};last={}
    for x in recs(1):
        if x[0]=='reclaim':L.put_reclaim(x[1]);F.put_reclaim(x[1])
        else:
            L.put(x[1],x[2]);F.put(x[1],x[2])
            if x[1] or x[2]:nrec+=1
        if r.random()<0.05:
            mv+=1;k=r.randrange(2);i=r.randrange(75) if k==0 else 0
            L.put_mark(k,i,mv);before[mv]=nrec;last[k,i]=mv
        if r.random()<0.01:time.sleep(0.002)
    L.publish(dict(rank=0,a=1));t=time.time();st={}
    while time.time()-t<30 and 1 not in st:st=L.peer_stats();time.sleep(0.05)
    q.put((st,before,last));time.sleep(1);L.close()

def follower(port,q):
    R=ON.NetOpLog(('127.0.0.1',port),writer=False,rank=1,nfollow=1,wait_s=60);r=random.Random(3);got=[];t=time.time();g=Gate();snap=[];pos_ok=True
    while time.time()-t<20:
        got+=R.get(g if r.random()<0.5 else None)
        for k,m in R.marks.items():
            for i in range(m.shape[0]):
                if m[i,0]:snap.append((int(m[i,0]),len(got)));pos_ok&=(R.gen,R.off)>=(int(m[i,1]),int(m[i,2]))
        time.sleep(r.random()*0.003)
        if len(got)>=sum(1 for x in recs(1) if x[0]=='ops' and (x[1] or x[2])):break
    time.sleep(0.5);got+=R.get()
    fin={(k,i):int(m[i,0]) for k,m in R.marks.items() for i in range(m.shape[0]) if m[i,0]}
    R.publish(dict(rank=1,b=2));q.put((got,snap,pos_ok,fin));time.sleep(2)

if __name__=='__main__':
    port=29611+os.getpid()%1000;q1=mp.Queue();q2=mp.Queue()
    with tempfile.TemporaryDirectory() as d:
        path=d+'/log.bin'
        pl=mp.Process(target=leader,args=(port,path,q1));pf=mp.Process(target=follower,args=(port,q2))
        pl.start();time.sleep(0.3);pf.start()
        (got,snap,pos_ok,fin),(st,before,last)=q2.get(timeout=60),q1.get(timeout=60);pf.join();pl.join()
        ref=[];R=OL.OpLog(path,writer=False)
        while True:
            x=R.get()
            if not x:break
            ref+=x
    ok=got==ref
    print(f'{"ok  " if ok else "FAIL"} stream records == file records ({len(got)} vs {len(ref)}, {sum(len(u)+len(v) for u,v in ref)} ops)')
    ok2=st.get(0)=={'rank':0,'a':1} and st.get(1)=={'rank':1,'b':2}
    print(f'{"ok  " if ok2 else "FAIL"} peer_stats {st}')
    ok3=pos_ok and all(n>=before[v] for v,n in snap) and fin==last and len(snap)>0
    print(f'{"ok  " if ok3 else "FAIL"} markers: {len(before)} sent, {len(snap)} follower checks, ops-before-marker and stream position hold, final marks == last sent')
    sys.exit(0 if ok and ok2 and ok3 else 1)
