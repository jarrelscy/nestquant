import torch;torch.cuda.set_per_process_memory_fraction(12/80)
from exl3_moe import *;from routing import *
for K in [2,4]:
    G=EXL3Group(16,K);m=EXL3MoE([(G,0)],4)
    x=(torch.randn(4,6144,device='cuda')*0.05).half();sel,rw=routing_tensors(make_sel(4,range(16)))
    y=m(x,sel,rw).clone();r=m.dense_ref(x,sel,rw);print(K,'rel',((y-r).norm()/r.norm()).item(),r.norm().item())
G2=EXL3Group(8,2,seed=1);G4=EXL3Group(8,4,seed=2);m=EXL3MoE([(G2,0),(G4,8)],3)
x=(torch.randn(3,6144,device='cuda')*0.05).half();sel,rw=routing_tensors(make_sel(3,range(16)))
y=m(x,sel,rw).clone();r=m.dense_ref(x,sel,rw);print('mix rel',((y-r).norm()/r.norm()).item())
