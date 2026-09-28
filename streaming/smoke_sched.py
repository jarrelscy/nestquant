"""Scheduler + executor + engine + mailbox on real layers of one TP rank, driven by tb4 decode routing.
  smoke_sched.py ROOT REPACK LAYERS(a-b) [rank=0] [tokens=512] [nslot=0: 60 per layer] [check_every=8] [tp=4]
Each step, for every layer: captured graph [mailbox.apply, MoE layer] with that token's top-8 from the routing log,
then executor.poll -> scheduler.step(counts) -> executor.apply. Every check_every steps each layer's output is
compared with a reference layer (pass: rel err <= max(1e-4, 2 x ref-vs-ref)) set to the levels the table actually held (level column of the table, resident
level-4 planes). Also reports the measured level-4 route share and op latencies. The fixed set is resident at level 4
(never streamed); the floating_default start is loaded through the executor before the first token."""
import os,sys,json,glob,time,collections,random,torch,numpy as np
HERE=os.path.dirname(os.path.abspath(__file__));sys.path[:0]=[HERE,HERE+'/../sm120']
torch.cuda.set_per_process_memory_fraction(40/96)
import nqload as NQ,stream_engine as SE,scheduler as SC,executor as EX,fixed_set as FS
from moe import MoELayer,Mailbox,entry
root,rp=sys.argv[1],sys.argv[2];a0,b0=(sys.argv[3].split('-')+[sys.argv[3]])[:2];LAYERS=list(range(int(a0),int(b0)+1))
a=[int(x) for x in sys.argv[4:]]+[None]*5;rank=a[0] or 0;NTOK=a[1] or 512;NSLOT=a[2] or 60*len(LAYERS);CK=a[3] or 8;tp=a[4] or 4
dev='cuda';NE=256;NF=51;L0=3
rf=SE.RankFile(rp,rank);fx,_,_=FS.load(layers=LAYERS)
fj=json.load(open(HERE+'/../threads/22-boundary-experts/fixed_set.json'))
dflt={L:[int(x) for x in np.argsort(-np.where(np.isin(np.arange(NE),fx[L]),-1,np.array(fj['n_routed'][str(L)])))[:NF]] for L in LAYERS}
t=time.time();RL={L:NQ.RankLayer(root,L,rank,tp) for L in LAYERS};H,I=RL[LAYERS[0]].H,RL[LAYERS[0]].I
print(f'layers {LAYERS} rank {rank}/TP{tp}: loaded in {time.time()-t:.0f}s; slots {NSLOT} x {rf.rb} B = {NSLOT*rf.rb/2**30:.1f} GiB',flush=True)
for L in LAYERS:assert len(RL[L].experts)==NE,f'L{L} has {len(RL[L].experts)} experts'
def setlv(M,L,E,lv):ex=RL[L].ex[E];ex.signs=ex.sc[lv];M.set(E,ex,lv);ex.signs=ex.sc[2]
lays={};refs={}
for L in LAYERS:
    M=MoELayer(NE,H,I,Bmax=1);MB=Mailbox(M);R=MoELayer(NE,H,I,Bmax=1)
    for E in range(NE):lv=4 if E in fx[L] else 2;setlv(M,L,E,lv);setlv(R,L,E,lv)
    lays[L]=(M,MB,RL[L].ex);refs[L]=R
# routing trace: the longest tb4 request in the first segments
D='/data/Jarrel/routing_logs/glm5.3-arvq-v2';reqs=collections.defaultdict(list)
for f in sorted(glob.glob(D+'/seg-*.npz'))[:6]:
    z=np.load(f);key=z['step'].astype(np.int64)*4096+z['req'];u,inv,cnt=np.unique(key,return_inverse=True,return_counts=True);m=cnt[inv]<=16
    r=z['req'][m];ex=z['experts'][m];pos=z['positions'][m];ids=z['req_ids']
    for q in np.unique(r):k=r==q;reqs[str(ids[q])].append((pos[k],ex[k]))
best=max(reqs.values(),key=lambda v:sum(len(p) for p,_ in v))
p=np.concatenate([x for x,_ in best]);e=np.concatenate([y for _,y in best]);o=np.argsort(p,kind='stable');p,e=p[o],e[o];e=e[np.r_[p[1:]!=p[:-1],True]]
e=e[:NTOK];print(f'routing: {len(e)} decode tokens',flush=True)
S=SC.Scheduler(LAYERS,fx,dflt,rf.rb*tp,NE=NE,n_float=NF)
X=EX.RankExecutor(rf,lays,NSLOT,n_host=64,qd=8)
rng=torch.Generator(device=dev).manual_seed(rank)
xs={L:(torch.randn(1,H,device=dev,generator=rng)*0.05).half() for L in LAYERS}
sel={L:torch.zeros(1,8,dtype=torch.long,device=dev) for L in LAYERS};rw=torch.full((1,8),1/8,device=dev)
graphs={}
s_=torch.cuda.Stream()
for L in LAYERS:
    M,MB,_=lays[L];f=lambda M=M,MB=MB,L=L:(MB.apply(),M(xs[L],sel[L],rw))
    with torch.cuda.stream(s_):
        for _ in range(2):f()
    torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):f()
    graphs[L]=g
torch.cuda.synchronize()
def replay_all():
    for L in LAYERS:graphs[L].replay()
# floating_default start through the executor (not counted against the cap)
init=[(L,E) for L in LAYERS for E in dflt[L] if E not in fx[L]]
for L,E in init:S.state[S.li[L],E]=1
X.apply(init,[],S);t=time.time()
while X.busy() and time.time()-t<60:replay_all();torch.cuda.synchronize();X.poll(S)
print(f'  floating_default loaded: {len(init)} upgrades in {time.time()-t:.2f}s, {X.n_refused} refused',flush=True)
worst=0;floor=0;nerr=0;nchk=0;hot=[];tstep=[];nbad_lv=0
for ti in range(len(e)):
    t1=time.perf_counter()
    for L in LAYERS:sel[L].copy_(torch.as_tensor(e[ti][L],device=dev).view(1,8))
    replay_all();torch.cuda.current_stream().synchronize()
    tb={L:lays[L][0].table[:,0].cpu().numpy() for L in LAYERS}
    hot.append(np.mean([(tb[L][e[ti][L]]==4).mean() for L in LAYERS]))
    if ti%CK==0:
        for L in LAYERS:
            o=lays[L][0].out[:1].float().clone()
            for E in set(e[ti][L].tolist()):setlv(refs[L],L,E,int(tb[L][E]))
            y=refs[L](xs[L],sel[L],rw).float().clone();y2=refs[L](xs[L],sel[L],rw).float()
            er=((o-y).norm()/y.norm()).item();fl=((y2-y).norm()/y.norm()).item();worst=max(worst,er);floor=max(floor,fl);nchk+=1
            nerr+=er>max(1e-4,2*fl)
    X.poll(S)
    c=np.zeros((len(LAYERS),NE));np.add.at(c,(np.arange(len(LAYERS))[:,None],np.array([e[ti][L] for L in LAYERS])),1)
    ups,downs=S.step(c,1);X.apply(ups,downs,S);tstep.append(time.perf_counter()-t1)
t=time.time()
while X.busy() and time.time()-t<30:replay_all();torch.cuda.synchronize();X.poll(S)
# the scheduler's view must match the tables once everything settled
lv=S.level()
for L in LAYERS:
    tbl=lays[L][0].table[:,0].cpu().numpy();nbad_lv+=int((tbl!=lv[S.li[L]]).sum())
X.close();lat=np.array(X.lat)*1e3
ok=nerr==0 and nbad_lv==0 and X.n_failed==0
r=dict(layers=[LAYERS[0],LAYERS[-1]],rank=rank,tokens=len(e),nslot=NSLOT,checks=nchk,worst_rel=float(f'{worst:.3e}'),ref_floor=float(f'{floor:.3e}'),over_bound=int(nerr),level_mismatch_after_settle=nbad_lv,
       route_share=round(float(np.mean(hot)),4),ups=S.stats['ups'],downs=S.stats['downs'],deferred_steps=S.stats['deferred_steps'],
       refused=X.n_refused,failed=X.n_failed,op_p50_ms=round(float(np.percentile(lat,50)),3),op_p99_ms=round(float(np.percentile(lat,99)),3),
       host_step_ms_p50=round(float(np.percentile(tstep,50))*1e3,2),ok=ok)
print(json.dumps(r));print('SCHED SMOKE','PASS' if ok else 'FAIL')
