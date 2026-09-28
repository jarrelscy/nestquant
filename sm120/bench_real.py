"""NQ MoE layer on real weights (one TP4 rank of a fitted/downloaded layer), per batch size B (MTP ns = B-1) and
tiling config. Level 4 = fixed set (+ the next most routed experts up to N4 total, the streamed share); routing
sampled from the layer's real expert frequencies (thread 22 n_routed), consecutive tokens reuse each expert w.p. 0.4
(MTP verify window). R routings per timed call so weights come from DRAM, as in serving.
  bench_real.py ROOT L [rank=0] [N4=77] [Bs=1,2,3,4,5,6,8] [out.json]"""
import os,sys,json,random,itertools,torch,numpy as np
HERE=os.path.dirname(os.path.abspath(__file__));sys.path[:0]=[HERE,HERE+'/../streaming']
torch.cuda.set_per_process_memory_fraction(20/96)
import nqload as NQ,fixed_set as FS
from moe import MoELayer
from timing import bench,warmup
root,L=sys.argv[1],int(sys.argv[2]);a=sys.argv[3:]+[None]*4
rank=int(a[0] or 0);N4=int(a[1] or 77);Bs=[int(b) for b in (a[2] or '1,2,3,4,5,6,8').split(',')];out=a[3]
NE=256;R=8
RL=NQ.RankLayer(root,L,rank,4);H,I=RL.H,RL.I
fr=np.array(json.load(open(HERE+'/../threads/22-boundary-experts/fixed_set.json'))['n_routed'][str(L)],float);p=fr/fr.sum()
fx,_,_=FS.load(layers=[L]);l4=list(fx[L])+[int(e) for e in np.argsort(-fr) if e not in fx[L]][:max(0,N4-len(fx[L]))]
M=MoELayer(NE,H,I,Bmax=max(Bs))
for E in range(NE):
    lv=4 if E in l4 else 2;ex=RL.ex[E];ex.signs=ex.sc[lv];M.set(E,ex,lv);ex.signs=ex.sc[2]
share4=float(p[l4].sum())
def rows(B,rng):
    rs=[]
    for t in range(B):
        r=[e for e in rs[-1] if rng.random()<0.4] if t else []
        while len(r)<8:
            e=int(rng.choice(NE,p=p))
            if e not in r:r.append(e)
        rs.append(r)
    return rs
cgs=[[1,8,3],[1,8,4],[1,8,6],[1,8,12],[2,8,3],[2,8,6],[1,4,6],[1,4,12]]
cds=[[1,8,1],[1,8,2],[1,8,4],[2,8,2],[1,4,2],[1,4,4]]
warmup();res=[]
for B in Bs:
    rng=np.random.default_rng(B);rts=[]
    for i in range(R):
        s=torch.tensor(rows(B,rng),device='cuda');w=torch.rand(B,8,device='cuda')+0.1;rts.append((s,(w/w.sum(1,keepdim=True)).half()))
    x=(torch.randn(B,H,device='cuda')*0.05).half()
    fns={(tuple(cg),tuple(cd)):(lambda cg=cg,cd=cd:[M(x,s,w,cfg_gu=cg,cfg_dn=cd) for s,w in rts]) for cg,cd in itertools.product(cgs,cds)}
    med,_=bench(fns,blocks=12,repeats=1);k=min(med,key=med.get);dflt=((1,8,3),(1,8,2))
    r=dict(L=L,rank=rank,B=B,n4=len(l4),share4=round(share4,3),best_us=round(med[k]/R,1),cfg=k,default_us=round(med[dflt]/R,1))
    res.append(r);print(json.dumps(r),flush=True)
if out:json.dump(res,open(out,'w'),indent=1)
