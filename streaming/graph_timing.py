"""Graph safety timing (B6): MoE step time of captured graphs [mailbox.apply, MoE layer] x layers on one TP rank, with the
SSD upgrade engine idle vs streaming flat out. Blocks of steps alternate idle / streaming so drift cancels.
  graph_timing.py ROOT REPACK LAYERS(a-b) [rank=0] [B=4] [blocks=10] [steps_per_block=200] [ops_per_step=4] [tp=4]
While streaming, every step issues up to ops_per_step random upgrades (level-2 non-fixed experts) and downgrades the
oldest landed ones to keep the pool full; host poll/apply runs while the graphs execute, as in serving. Run one process
per GPU at the same time to load the SSD from all 4 ranks. Output: one JSON line (GPU us per step, idle vs streaming)."""
import os,sys,json,time,random,collections,torch,numpy as np
HERE=os.path.dirname(os.path.abspath(__file__));sys.path[:0]=[HERE,HERE+'/../sm120']
torch.cuda.set_per_process_memory_fraction(40/96)
import nqload as NQ,stream_engine as SE,executor as EX,fixed_set as FS
from moe import MoELayer,Mailbox
root,rp=sys.argv[1],sys.argv[2];a0,b0=(sys.argv[3].split('-')+[sys.argv[3]])[:2];LAYERS=list(range(int(a0),int(b0)+1))
a=[int(x) for x in sys.argv[4:]]+[None]*7
rank=a[0] or 0;B=a[1] or 4;NBLK=a[2] or 10;SPB=a[3] or 200;OPS=a[4] or 4;tp=a[5] or 4
dev='cuda';NE=256;NSLOT=60*len(LAYERS);rng=random.Random(rank)
rf=SE.RankFile(rp,rank);fx,_,_=FS.load(layers=LAYERS)
RL={L:NQ.RankLayer(root,L,rank,tp) for L in LAYERS};H,I=RL[LAYERS[0]].H,RL[LAYERS[0]].I
lays={}
for L in LAYERS:
    M=MoELayer(NE,H,I,Bmax=B);MB=Mailbox(M)
    for E in range(NE):lv=4 if E in fx[L] else 2;ex=RL[L].ex[E];ex.signs=ex.sc[lv];M.set(E,ex,lv);ex.signs=ex.sc[2]
    lays[L]=(M,MB,RL[L].ex)
class St:
    """minimal scheduler stand-in: tracks levels for the executor callbacks"""
    def __init__(s):s.st={};s.landed_q=collections.deque()
    def landed(s,L,E):s.st[L,E]=2;s.landed_q.append((L,E))
    def released(s,L,E):s.st.pop((L,E),None)
    def failed(s,L,E,read_error=False):s.st.pop((L,E),None)
S=St();X=EX.RankExecutor(rf,lays,NSLOT,n_host=64,qd=8)
x=(torch.randn(B,H,device=dev)*0.05).half();rw=torch.full((B,8),1/8,device=dev)
sel={L:torch.stack([torch.randperm(NE,device=dev)[:8] for _ in range(B)]) for L in LAYERS}
s_=torch.cuda.Stream()
def f():
    for L in LAYERS:M,MB,_=lays[L];MB.apply();M(x,sel[L],rw)
with torch.cuda.stream(s_):
    for _ in range(2):f()
torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
with torch.cuda.graph(g):f()
free_E={L:[E for E in range(NE) if E not in fx[L]] for L in LAYERS}
def stream_ops():
    ups=[];downs=[]
    for _ in range(OPS):
        L=rng.choice(LAYERS);E=rng.choice(free_E[L])
        if (L,E) not in S.st and len(S.st)<NSLOT:S.st[L,E]=1;ups.append((L,E))
    while len(S.st)>NSLOT-2*OPS and S.landed_q:
        k=S.landed_q.popleft()
        if S.st.get(k)==2:S.st[k]=3;downs.append(k)
    X.apply(ups,downs,S)
ev=[(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)) for _ in range(SPB)]
res={0:[],1:[]};wall=0.0
for blk in range(2*NBLK):
    mode=blk%2;tb=time.time()
    for i in range(SPB):
        for L in LAYERS:sel[L].copy_(torch.stack([torch.randperm(NE,device=dev)[:8] for _ in range(B)]))
        ev[i][0].record();g.replay();ev[i][1].record()
        X.poll(S)
        if mode:stream_ops()
        ev[i][1].synchronize()
    torch.cuda.synchronize()
    if mode:wall+=time.time()-tb
    if blk>=2:res[mode]+=[s.elapsed_time(e)*1e3 for s,e in ev]
t=time.time()
while X.busy() and time.time()-t<30:g.replay();torch.cuda.synchronize();X.poll(S)
X.close();lat=np.array(X.lat)*1e3
r0,r1=np.array(res[0]),np.array(res[1])
out=dict(layers=[LAYERS[0],LAYERS[-1]],rank=rank,B=B,steps=len(r0),ops_per_step=OPS,
         idle_us=dict(mean=round(r0.mean(),1),p50=round(float(np.median(r0)),1),p99=round(float(np.percentile(r0,99)),1)),
         stream_us=dict(mean=round(r1.mean(),1),p50=round(float(np.median(r1)),1),p99=round(float(np.percentile(r1,99)),1)),
         regress_pct=round((r1.mean()/r0.mean()-1)*100,2),
         upgrades=len(lat),GBps=round(len(lat)*rf.rb/wall/1e9,2),
         op_p50_ms=round(float(np.percentile(lat,50)),2) if len(lat) else None,op_p99_ms=round(float(np.percentile(lat,99)),2) if len(lat) else None,
         failed=X.n_failed)
print(json.dumps(out))
