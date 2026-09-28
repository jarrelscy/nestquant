"""TP8 shard shape (I=256 per GPU, H=6144, 256 experts resident), cold, recency routing: NQ vs EXL3 coop.
NQ config = best of {tune2_I256.json per-stage configs (joint + per-level), a small fixed sweep} on the same run.
EXL3 launch options come from the environment (EXL3_MOE_COOP_KSPLIT / _WIDE, read once per process; the scratch is
sized for any k-split); the tag is recorded. Appends to bench_shard.json."""
import torch,random,json,itertools,os;torch.cuda.set_per_process_memory_fraction(12/80)
from moe import *;from exl3_moe import *;from routing import *;from timing import bench,warmup
H,I=6144,256;NP=256;R=8
tag=dict(KSPLIT=os.environ.get('EXL3_MOE_COOP_KSPLIT','-'),WIDE=os.environ.get('EXL3_MOE_COOP_WIDE','-'))
tune=json.load(open('tune2_I256.json'))['default']
ex=[Expert(H,I,seed=i) for i in range(NP)];exf=[Expert(H,I,seed=1000+i,rk_gu=1,rk_dn=2) for i in range(NP)]
Ls={}
for lv in (2,4,'4f'):
    L=MoELayer(NP,H,I);[L.set(e,(exf if lv=='4f' else ex)[e],4 if lv=='4f' else lv) for e in range(NP)];Ls[lv]=L
G={2:EXL3Group(NP,2,H=H,I=I,seed=1),4:EXL3Group(NP,4,H=H,I=I,seed=2)}
fixed_g=[[1,8,3],[1,8,6],[2,8,3],[1,4,6]];fixed_d=[[1,8,2],[2,8,1],[1,4,2]]
warmup();res=[]
for B in [1,2,3,4]:
    rng=random.Random(B);rts=[];prev=()
    for i in range(R):
        rows=make_sel(B,range(NP),'recency',rng=rng,avoid=prev);prev=[e for r in rows for e in r];rts.append(routing_tensors(rows,seed=i))
    x=(torch.randn(B,H,device='cuda')*0.05).half()
    tg,td=tune[f'gu|B{B}'],tune[f'dn|B{B}']
    uniq=lambda l:[list(t) for t in dict.fromkeys(tuple(c) for c in l)]
    cgs=uniq([tg['cfg'],tg['cfg2'],tg['cfg4']]+fixed_g);cds=uniq([td['cfg'],td['cfg2'],td['cfg4']]+fixed_d)
    fns={}
    for lv in (2,4):
        m=EXL3MoE([(G[lv],0)],B,H=H,I=I,smax=256);fns[('exl3',lv)]=(lambda m=m:[m(x,s,w) for s,w in rts])
    for lv in (2,4,'4f'):
        for cg,cd in itertools.product(cgs,cds):
            fns[('nq',lv,tuple(cg),tuple(cd))]=(lambda L=Ls[lv],cg=cg,cd=cd:[L(x,s,w,cfg_gu=cg,cfg_dn=cd) for s,w in rts])
    med,_=bench(fns,blocks=12,repeats=1)
    for lv in (2,4,'4f'):
        k=min([k for k in med if k[0]=='nq' and k[1]==lv],key=lambda k:med[k]);te=med[('exl3',4 if lv=='4f' else lv)]
        r=dict(B=B,level=lv,nq_us=round(med[k]/R,1),exl3_us=round(te/R,1),exl3_opts=tag,cfg=k[2:],speedup=round(te/med[k],2))
        res.append(r);print(r,flush=True)
old=json.load(open('bench_shard.json')) if os.path.exists('bench_shard.json') and os.environ.get('APPEND') else []
json.dump(old+res,open('bench_shard.json','w'),indent=1)
