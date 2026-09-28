"""TP4 shard shape (I=512 per GPU, H=6144, 256 experts resident), cold L2 between calls, recency routing:
NestQuant grouped MoE vs EXL3 exl3_moe_coop, at 2 and 4 bit and a 50/50 mix. NQ config = best of a sweep per (B, level)."""
import torch,random,json,itertools,sys,os;torch.cuda.set_per_process_memory_fraction(14/96)
from moe import *;from exl3_moe import *;from routing import *;from timing import bench,warmup
H,I=6144,512;NP=256;R=8
out=sys.argv[1] if len(sys.argv)>1 else 'results/bench_tp4.json'
ex=[Expert(H,I,seed=i,rk_gu=6,rk_dn=7) for i in range(NP)]   # production 4-bit: gate/up K 1.9375, down K 2.3125 (4.0846 bpw)
Ls={}
for lv in (2,4,'mix'):
    L=MoELayer(NP,H,I);[L.set(e,ex[e],(lv if lv!='mix' else (2 if e<NP//2 else 4))) for e in range(NP)];Ls[lv]=L
G={2:EXL3Group(NP,2,H=H,I=I,seed=1),4:EXL3Group(NP,4,H=H,I=I,seed=2)}
Gm=[EXL3Group(NP//2,2,H=H,I=I,seed=3),EXL3Group(NP//2,4,H=H,I=I,seed=4)]
cgs=[[1,8,3],[1,8,6],[1,8,4],[2,8,3],[1,4,6],[2,4,6],[1,8,12],[2,8,6],[1,4,12]]
cds=[[1,8,1],[1,8,2],[1,8,4],[2,8,1],[2,8,2],[1,4,1],[1,4,2],[1,4,4]]
warmup();res=[]
for B in [1,2,3,4]:
    rng=random.Random(B);rts=[];prev=()
    for i in range(R):
        rows=make_sel(B,range(NP),'recency',rng=rng,avoid=prev);prev=[e for r in rows for e in r];rts.append(routing_tensors(rows,seed=i))
    # the mixed layer: route half the picks into each half so the mix is ~50/50 by picks
    x=(torch.randn(B,H,device='cuda')*0.05).half()
    fns={}
    for lv in (2,4):
        m=EXL3MoE([(G[lv],0)],B,H=H,I=I,smax=2*B*8);fns[('exl3',lv)]=(lambda m=m:[m(x,s,w) for s,w in rts])
    mm=EXL3MoE([(Gm[0],0),(Gm[1],NP//2)],B,H=H,I=I,smax=2*B*8);fns[('exl3','mix')]=(lambda m=mm:[m(x,s,w) for s,w in rts])
    for lv in (2,4,'mix'):
        for cg,cd in itertools.product(cgs,cds):
            fns[('nq',lv,tuple(cg),tuple(cd))]=(lambda L=Ls[lv],cg=cg,cd=cd:[L(x,s,w,cfg_gu=cg,cfg_dn=cd) for s,w in rts])
    med,_=bench(fns,blocks=12,repeats=1)
    for lv in (2,4,'mix'):
        k=min([k for k in med if k[0]=='nq' and k[1]==lv],key=lambda k:med[k])
        r=dict(B=B,level=lv,nq_us=round(med[k]/R,1),exl3_us=round(med[('exl3',lv)]/R,1),cfg=k[2:],speedup=round(med[('exl3',lv)]/med[k],2))
        r['exl3_ksplit']=os.environ.get('EXL3_MOE_COOP_KSPLIT','1');res.append(r);print(r,flush=True)
json.dump(res,open(out,'w'),indent=1)
