"""Replay tb4 decode routing through scheduler.Scheduler token by token (one stream). Upgrades land `lag` steps after
they are issued. Reports route share on level-4 experts (from token 4096, as select_sweep, and from token 0 with the
floating_default start), SSD GB/s at 111 tok/s, deferred and big-step counts.
  sched_sim.py [lag=1] [cap_GBps=6] [half_life=512] [max_seqs]"""
import sys,json,glob,collections,time,numpy as np
sys.path.insert(0,'/data/Jarrel/nestquant/streaming');import fixed_set,scheduler
D='/data/Jarrel/routing_logs/glm5.3-arvq-v2';NE=256;L0=3;NL=75;NF=51;MB=9.97e6
a=[float(x) for x in sys.argv[1:]]+[None]*4;LAG=int(a[0] or 1);CAP=a[1] or 6.0;HL=a[2] or 512;MAXS=int(a[3] or 1000)
fx,_,_=fixed_set.load();layers=list(range(L0,L0+NL))
fj=json.load(open('/data/Jarrel/nestquant/threads/22-boundary-experts/fixed_set.json'))
fixed=np.zeros((NL,NE),bool)
for l in range(NL):fixed[l,fx[l+L0]]=True
dflt={l+L0:[int(x) for x in np.argsort(-np.where(fixed[l],-1,np.array(fj['n_routed'][str(l+L0)])))[:NF]] for l in range(NL)}
reqs=collections.defaultdict(list);tot=0
for f in sorted(glob.glob(D+'/seg-*.npz')):
    z=np.load(f);key=z['step'].astype(np.int64)*4096+z['req'];u,inv,cnt=np.unique(key,return_inverse=True,return_counts=True);m=cnt[inv]<=16
    if not m.any():continue
    r=z['req'][m];ex=z['experts'][m][:,L0:L0+NL];pos=z['positions'][m];ids=z['req_ids']
    for q in np.unique(r):k=r==q;reqs[str(ids[q])+f.split('-')[2]].append((pos[k],ex[k]))
    tot+=int(m.sum())
    if tot>=1_500_000:break
seqs=[]
for v in reqs.values():
    p=np.concatenate([a for a,_ in v]);e=np.concatenate([b for _,b in v]);o=np.argsort(p,kind='stable');p,e=p[o],e[o];e=e[np.r_[p[1:]!=p[:-1],True]]
    if len(e)>=4096+256:seqs.append(e)
seqs=seqs[:MAXS];t0=time.time()
hot0=[];hot4=[];nb=0;ntok=0;st=collections.Counter()
for e in seqs:
    S=scheduler.Scheduler(layers,fx,dflt,MB,NE=NE,n_float=NF,half_life=HL,refresh=64,cap_GBps=CAP,predictor='ema')
    # initial floating_default loads before the first token (not counted against the cap)
    for i,x in zip(*np.nonzero(S.want)):S.state[i,x]=2
    inflight=collections.deque();ar=np.arange(NL)[:,None]
    for t in range(len(e)):
        lv=S.level();h=(lv[ar,e[t]]==4).mean()
        hot0.append(h)
        if t>=4096:hot4.append(h)
        c=np.zeros((NL,NE));np.add.at(c,(ar,e[t]),1)
        ups,downs=S.step(c,1);inflight.append((ups,downs))
        if len(inflight)>LAG:
            u,d=inflight.popleft()
            for L,x in u:S.landed(L,x)
            for L,x in d:S.released(L,x)
    nb+=S.stats['bytes'];ntok+=len(e);st.update({k:v for k,v in S.stats.items() if k!='bytes'})
r=dict(lag=LAG,cap_GBps=CAP,half_life=HL,seqs=len(seqs),tokens=ntok,route_share_from4096=round(float(np.mean(hot4)),4),
       route_share_all=round(float(np.mean(hot0)),4),GBps_111=round(nb/ntok*111/1e9,2),**st,secs=round(time.time()-t0))
print(json.dumps(r))
open('/data/Jarrel/nestquant/streaming/results/sched_sim.jsonl','a').write(json.dumps(r)+'\n')
