"""TP8 shard shape (I=256 per GPU, H=6144, 256 experts resident), cold, recency routing: NQ vs EXL3 coop.
NQ config = best of a small sweep per (B, level) chosen on the same run (tune for shard shape not done separately)."""
import torch,random,json,itertools;torch.cuda.set_per_process_memory_fraction(12/80)
from moe import *;from exl3_moe import *;from routing import *;from timing import bench,warmup
H,I=6144,256;NP=256;R=8
ex=[Expert(H,I,seed=i) for i in range(NP)]
Ls={}
for lv in (2,4):
    L=MoELayer(NP,H,I);[L.set(e,ex[e],lv) for e in range(NP)];Ls[lv]=L
G={2:EXL3Group(NP,2,H=H,I=I,seed=1),4:EXL3Group(NP,4,H=H,I=I,seed=2)}
cgs=[[1,8,3],[1,8,6],[2,8,3],[1,4,6],[2,4,6]];cds=[[1,8,2],[1,4,2],[2,8,1],[1,8,1],[1,4,1]]
warmup();res=[]
for B in [1,2,3,4]:
    rng=random.Random(B);rts=[];prev=()
    for i in range(R):
        rows=make_sel(B,range(NP),'recency',rng=rng,avoid=prev);prev=[e for r in rows for e in r];rts.append(routing_tensors(rows,seed=i))
    x=(torch.randn(B,H,device='cuda')*0.05).half()
    fns={}
    for lv in (2,4):
        m=EXL3MoE([(G[lv],0)],B,H=H,I=I);fns[('exl3',lv)]=(lambda m=m:[m(x,s,w) for s,w in rts])
        for cg,cd in itertools.product(cgs,cds):
            fns[('nq',lv,tuple(cg),tuple(cd))]=(lambda L=Ls[lv],cg=cg,cd=cd:[L(x,s,w,cfg_gu=cg,cfg_dn=cd) for s,w in rts])
    med,_=bench(fns,blocks=12,repeats=1)
    for lv in (2,4):
        k=min([k for k in med if k[0]=='nq' and k[1]==lv],key=lambda k:med[k])
        r=dict(B=B,level=lv,nq_us=round(med[k]/R,1),exl3_us=round(med[('exl3',lv)]/R,1),cfg=k[2:],speedup=round(med[('exl3',lv)]/med[k],2))
        res.append(r);print(r,flush=True)
json.dump(res,open('bench_shard.json','w'),indent=1)
