import torch,random;torch.cuda.set_per_process_memory_fraction(12/80)
from exl3_moe import *;from routing import *;from timing import bench
NP=72;R=8
G2=EXL3Group(NP,2,seed=1);G4=EXL3Group(NP,4,seed=2)
Gm2=EXL3Group(NP//2,2,seed=3);Gm4=EXL3Group(NP//2,4,seed=4)
print('exl3 bytes/expert',G2.bytes_per_expert,G4.bytes_per_expert)
for mode in ['recency','distinct']:
  for B in [1,2,3,4]:
    rng=random.Random(B);rts=[];prev=()
    for i in range(R):
        rows=make_sel(B,range(NP),mode,rng=rng,avoid=prev);prev=[e for r in rows for e in r];rts.append(routing_tensors(rows,seed=i))
    x=(torch.randn(B,6144,device='cuda')*0.05).half()
    ms={'2b':EXL3MoE([(G2,0)],B),'4b':EXL3MoE([(G4,0)],B),'mix':EXL3MoE([(Gm2,0),(Gm4,NP//2)],B)}
    fns={k:(lambda m=m:[m(x,s,w) for s,w in rts]) for k,m in ms.items()}
    med,_=bench(fns,blocks=30,repeats=1)
    print(mode,B,{k:round(v/R,1) for k,v in med.items()},flush=True)
