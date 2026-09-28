"""Correctness: grouped MoE kernel vs dense decode reference, mixed levels 0/2/3/4, B=1..4, recency + forced duplicates."""
import torch,random,sys;torch.cuda.set_per_process_memory_fraction(12/80)
from moe import *;from routing import *
H,I=6144,int(sys.argv[1]) if len(sys.argv)>1 else 2048
n3g,n3d=H//128//2,I//128//2
NE=12
ex=[Expert(H,I,n3g,n3d,seed=100+i) for i in range(NE)]
levels=[2,3,4,2,3,4,2,4,3,4,2,0]
L=MoELayer(NE,H,I,n3g,n3d)
for e in range(NE):L.set(e,ex[e],levels[e])
rng=random.Random(0)
for B in [1,2,3,4]:
    for mode in ['recency','distinct','same']:
        if mode=='same':rows=[make_sel(1,range(NE),rng=rng)[0]]*B
        else:rows=make_sel(B,range(NE),'recency' if mode=='recency' else 'distinct',p=0.5,rng=rng) if not(mode=='distinct' and B*8>NE) else make_sel(B,range(NE),'recency',p=0.0,rng=rng)
        sel,rw=routing_tensors(rows,seed=B)
        x=(torch.randn(B,H,device='cuda')*0.05).half()
        y=L(x,sel,rw).clone();y2=L(x,sel,rw).clone()
        r=moe_ref(ex,levels,x,sel,rw)
        print(B,mode,'distinct experts',len(set(sum(rows,[]))),'rel err',round(((y-r).norm()/r.norm()).item(),6),'repeat bitwise',torch.equal(y,y2),flush=True)
print('ws clean',L.acc_gu.abs().max().item(),L.acc_d.abs().max().item(),L.cnt_gu.abs().sum().item(),L.cnt_d.abs().sum().item())
