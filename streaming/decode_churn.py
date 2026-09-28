"""SSD demand and 4-bit coverage of the decode scheduler on tb4 routing logs (decode tokens only).
Per request: fixed set (26/layer) + floating top-51 non-fixed by count over the last 1024 decode tokens, refreshed every
64 tokens. The set picked from [s-1024, s) serves tokens [s+64*lag, s+64*lag+64): lag 1 = reads overlap the next window.
Fixed sets: 'reap' = thread-22 capture W=50; 'usage' = tb4 decode usage top-26 from the OTHER half of requests (held out).
Metrics: route share (top-8 slots on hot experts), activated-hot (distinct experts routed in the 64-token window that
were hot), new floating experts per refresh (summed over 75 layers) and SSD GB/s at 9.97 MB/expert."""
import numpy as np,glob,collections,sys,json
D='/data/Jarrel/routing_logs/glm5.3-arvq-v2';NE=256;L0=3;NL=75;K=26;NF=51;WIN=1024;REF=64;MB=9.97e6;LAGS=(0,1,2)
cap=int(sys.argv[1]) if len(sys.argv)>1 else 1_500_000
reqs=collections.defaultdict(list);tot=0
for f in sorted(glob.glob(D+'/seg-*.npz')):
    z=np.load(f);st=z['step'];rq=z['req'];key=st.astype(np.int64)*4096+rq;u,inv,cnt=np.unique(key,return_inverse=True,return_counts=True)
    m=cnt[inv]<=16
    if not m.any():continue
    ids=z['req_ids'];ex=z['experts'][m][:,L0:L0+NL];pos=z['positions'][m];r=rq[m]
    for q in np.unique(r):k=r==q;reqs[str(ids[q])+f.split('-')[2]].append((pos[k],ex[k]))
    tot+=int(m.sum())
    if tot>=cap:break
seqs=[]
for v in reqs.values():
    p=np.concatenate([a for a,_ in v]);e=np.concatenate([b for _,b in v]);o=np.argsort(p,kind='stable');p,e=p[o],e[o]
    e=e[np.r_[True,p[1:]!=p[:-1]]]
    if len(e)>=WIN+3*REF:seqs.append(e)
half=[np.zeros((NL,NE),np.int64) for _ in range(2)]
for i,e in enumerate(seqs):
    for l in range(NL):half[i%2][l]+=np.bincount(e[:,l].ravel(),minlength=NE)
capj=json.load(open('/data/Jarrel/nestquant/threads/22-boundary-experts/reap_weighted_19-capture.json'))
def mask(sets):
    m=np.zeros((NL,NE),bool)
    for l in range(NL):m[l,sets[l]]=True
    return m
reap=mask([capj[str(l+L0)]['50']['top'][:K] for l in range(NL)])
usage=[mask([np.argsort(-half[1-h][l])[:K] for l in range(NL)]) for h in (0,1)]   # held out: built from the other half
res={}
for name in ('reap','usage'):
    new=[];rs={l:[] for l in LAGS};act={l:[] for l in LAGS};fxs=[];fxa=[];ntok=0
    for i,e in enumerate(seqs):
        fixed=reap if name=='reap' else usage[i%2]
        oh=np.zeros((len(e),NL,NE),np.int32);np.add.at(oh,(np.arange(len(e))[:,None,None],np.arange(NL)[None,:,None],e),1)
        cs=np.concatenate([np.zeros((1,NL,NE),np.int64),np.cumsum(oh,0)]);prev=None
        for s in range(WIN,len(e)-REF+1,REF):
            c=np.where(fixed,-1,cs[s]-cs[s-WIN]);fl=np.zeros_like(fixed);np.put_along_axis(fl,np.argsort(-c,1)[:,:NF],True,1)
            if prev is not None:new.append(int((fl&~prev).sum()))
            prev=fl;hot=fl|fixed
            for lag in LAGS:
                a=s+lag*REF
                if a+REF>len(e):continue
                w=cs[a+REF]-cs[a];used=w>0
                rs[lag].append((w*hot).sum()/w.sum());act[lag].append((used&hot).sum()/used.sum())
            w=cs[s+REF]-cs[s];used=w>0;fxs.append((w*fixed).sum()/w.sum());fxa.append((used&fixed).sum()/used.sum());ntok+=REF
    n=np.array(new);g=float(n.mean()*MB/1e9)
    r=dict(decode_tokens=ntok,requests=len(seqs),fixed_route_share=float(np.mean(fxs)),fixed_activated_share=float(np.mean(fxa)),
      **{f'route_share_lag{l}':float(np.mean(rs[l])) for l in LAGS},**{f'activated_hot_lag{l}':float(np.mean(act[l])) for l in LAGS},
      new_per_refresh=float(n.mean()),new_p95=float(np.percentile(n,95)),GB_per_refresh=g,**{f'GBps_at_{t}tps':round(g*t/REF,2) for t in (60,111,148,185,214)})
    res[name]=r;print(name,json.dumps({k:round(v,4) if isinstance(v,float) else v for k,v in r.items()}),flush=True)
json.dump(res,open('/data/Jarrel/nestquant/streaming/results/decode_churn.json','w'),indent=1)
