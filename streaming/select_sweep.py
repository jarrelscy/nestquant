"""Floating-set selection sweep on tb4 decode (final fixed set + 51 floating, 1-window lag). Scores: box window of W
tokens, or exponentially decayed counts with half-life H tokens; refresh every R tokens. Reports route share on hot
experts, activated-hot, and SSD GB/s at 111 tok/s (9.97 MB per new expert)."""
import numpy as np,glob,collections,sys,json
sys.path.insert(0,'/data/Jarrel/nestquant/streaming');import fixed_set
D='/data/Jarrel/routing_logs/glm5.3-arvq-v2';NE=256;L0=3;NL=75;NF=51;MB=9.97e6
fx,_,_=fixed_set.load();fixed=np.zeros((NL,NE),bool)
for l in range(NL):fixed[l,fx[l+L0]]=True
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
cfgs=[('box',W,64) for W in (256,512,1024,2048,4096)]+[('ema',H,64) for H in (128,256,512,1024)]+[('box',1024,32),('box',1024,16),('ema',256,32),('ema',256,16)]
res=[]
for kind,P,R in cfgs:
    rs=[];ah=[];new=[];ntok=0
    for e in seqs:
        n=len(e);oh=np.zeros((n,NL,NE),np.float32);np.add.at(oh,(np.arange(n)[:,None,None],np.arange(NL)[None,:,None],e),1)
        cs=np.concatenate([np.zeros((1,NL,NE)),np.cumsum(oh,0)])
        if kind=='ema':
            a=0.5**(1/P);sc=np.zeros((NL,NE));ema=np.zeros((n+1,NL,NE),np.float32)
            for i in range(n):sc=sc*a+oh[i];ema[i+1]=sc
        prev=None
        for s in range(4096,n-2*R+1,R):
            c=(cs[s]-cs[max(0,s-P)]) if kind=='box' else ema[s].astype(np.float64)
            c=np.where(fixed,-1,c);fl=np.zeros_like(fixed);np.put_along_axis(fl,np.argsort(-c,1)[:,:NF],True,1)
            if prev is not None:new.append(int((fl&~prev).sum()))
            prev=fl;hot=fl|fixed;w=cs[s+2*R]-cs[s+R];used=w>0
            rs.append((w*hot).sum()/w.sum());ah.append((used&hot).sum()/used.sum());ntok+=R
    g=np.mean(new)*MB/1e9;r=dict(score=kind,param=P,refresh=R,route_share=round(float(np.mean(rs)),4),activated_hot=round(float(np.mean(ah)),4),GBps_111=round(g*111/R,2))
    res.append(r);print(r,flush=True)
json.dump(res,open('/data/Jarrel/nestquant/streaming/results/select_sweep.json','w'),indent=1)
