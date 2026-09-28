"""EXL3 1.5.1 exl3_moe_coop launch-option sweep at the shard shape (one config per process: the .so reads
EXL3_MOE_COOP_KSPLIT / EXL3_MOE_COOP_WIDE once). Same cold recency routings as bench_shard.py. Appends to
exl3_sweep_I{I}.json. Checks the output against EXL3's own dense reference once per (B, K)."""
import torch,random,json,os;torch.cuda.set_per_process_memory_fraction(12/80)
from exl3_moe import *;from routing import *;from timing import bench,warmup
H=6144;I=int(os.environ.get('I',256));NP=256 if I<1024 else 72;R=8
tag=dict(KSPLIT=os.environ.get('EXL3_MOE_COOP_KSPLIT','-'),WIDE=os.environ.get('EXL3_MOE_COOP_WIDE','-'))
G={2:EXL3Group(NP,2,H=H,I=I,seed=1),4:EXL3Group(NP,4,H=H,I=I,seed=2)}
warmup();res=[]
for B in [1,2,3,4]:
    rng=random.Random(B);rts=[];prev=()
    for i in range(R):
        rows=make_sel(B,range(NP),'recency',rng=rng,avoid=prev);prev=[e for r in rows for e in r];rts.append(routing_tensors(rows,seed=i))
    x=(torch.randn(B,H,device='cuda')*0.05).half()
    fns={};ms={}
    for lv in (2,4):
        m=EXL3MoE([(G[lv],0)],B,H=H,I=I,smax=256);ms[lv]=m
        s,w=rts[0];y=m(x,s,w).clone();r=m.dense_ref(x,s,w);err=((y-r).norm()/r.norm()).item()
        fns[lv]=(lambda m=m:[m(x,s,w) for s,w in rts])
        res.append(dict(B=B,K=lv,**tag,relerr=err))
    med,_=bench(fns,blocks=12,repeats=1)
    for lv in (2,4):
        res[-2+(lv==4)]['us']=med[lv]/R;print(res[-2+(lv==4)],flush=True)
fn=f'exl3_sweep_I{I}.json';old=json.load(open(fn)) if os.path.exists(fn) else []
json.dump(old+res,open(fn,'w'),indent=1)
