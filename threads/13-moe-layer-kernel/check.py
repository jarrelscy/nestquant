"""Correctness: grouped MoE kernel vs dense decode reference, mixed levels 0/2/4 (+ 4 in 128x128 mask mode), B=1..4, recency + forced duplicates."""
import torch,random,sys;torch.cuda.set_per_process_memory_fraction(12/80)
from moe import *;from routing import *
H,I=6144,int(sys.argv[1]) if len(sys.argv)>1 else 2048
nmg,nmd=H//128//2,I//128//2
NE=12
masked=[1,4,8]                        # these experts carry a 50% 128x128-block P4 mask
ex=[Expert(H,I,seed=100+i,nm_gu=nmg if i in masked else None,nm_dn=nmd if i in masked else None) for i in range(NE)]
levels=[2,4,4,2,4,4,2,4,4,4,2,0]
L=MoELayer(NE,H,I,nmg,nmd)
for e in range(NE):L.set(e,ex[e],levels[e])
rng=random.Random(0)
for B in [1,2,3,4]:
    for mode in ['recency','distinct','same']:
        if mode=='same':rows=[make_sel(1,range(NE),rng=rng)[0]]*B
        else:rows=make_sel(B,range(NE),'recency' if mode=='recency' else 'distinct',p=0.5,rng=rng) if not(mode=='distinct' and B*8>NE) else make_sel(B,range(NE),'recency',p=0.0,rng=rng)
        sel,rw=routing_tensors(rows,seed=B)
        x=(torch.randn(B,H,device='cuda')*0.05).half()
        r=moe_ref(ex,levels,x,sel,rw)
        for cg,cd in ([([1,8,3],[1,8,2]),([2,8,4,1],[2,8,4,1]),([1,4,6,2],[1,8,2,1])] if I>=1024 else [([1,8,3],[1,8,2]),([2,8,4],[1,4,2]),([1,4,6,2],[2,8,1,1])]):
            y=L(x,sel,rw,cfg_gu=cg,cfg_dn=cd).clone();y2=L(x,sel,rw,cfg_gu=cg,cfg_dn=cd).clone()
            print(B,mode,cg,cd,'distinct experts',len(set(sum(rows,[]))),'rel err',round(((y-r).norm()/r.norm()).item(),6),'repeat bitwise',torch.equal(y,y2),flush=True)
print('ws clean',L.acc_gu.abs().max().item(),L.acc_d.abs().max().item(),L.cnt_gu.abs().sum().item(),L.cnt_d.abs().sum().item(),L.wq.abs().sum().item())
