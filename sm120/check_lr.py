"""Correctness of the T12 low-rank plane in the kernel: experts with random lr (r_gu, r_dn in 0..4, incl. gate-only and
down-only), mixed levels 0/2/4 (+ mask mode), B=1..4, persistent and grid kernels, vs Expert.ref (dense decode + lr)."""
import torch,random,sys;torch.cuda.set_per_process_memory_fraction(12/80)
from moe import *;from routing import *
H,I=6144,int(sys.argv[1]) if len(sys.argv)>1 else 2048
nmg,nmd=H//128//2,I//128//2
NE=10;masked=[1,4]
ranks=[(0,0),(1,0),(0,1),(2,3),(4,4),(3,1),(4,0),(0,4),(2,2),(1,1)]
ex=[Expert(H,I,seed=300+i,nm_gu=nmg if i in masked else None,nm_dn=nmd if i in masked else None) for i in range(NE)]
for i,(rg,rd) in enumerate(ranks):
    g=torch.Generator().manual_seed(900+i);rn=lambda *s:(torch.randn(*s,generator=g)*0.3).half().cuda()
    un=lambda r,n:torch.nn.functional.normalize(torch.randn(r,n,generator=g),dim=1).half().cuda()
    ex[i].set_lr(un(rg,H),rn(rg,I),rn(rg,I),rn(rg,I),rn(rg,I),un(rd,I),rn(rd,H),rn(rd,H))
levels=[4,4,2,4,4,2,4,4,2,0]
L=MoELayer(NE,H,I,nmg,nmd)
for e in range(NE):L.set(e,ex[e],levels[e])
rng=random.Random(0);worst=0
for B in [1,2,3,4]:
    for mode in ['recency','same']:
        rows=[make_sel(1,range(NE),rng=rng)[0]]*B if mode=='same' else make_sel(B,range(NE),'recency',p=0.5,rng=rng)
        sel,rw=routing_tensors(rows,seed=B);x=(torch.randn(B,H,device='cuda')*0.05).half()
        r=moe_ref(ex,levels,x,sel,rw)
        for cg,cd in ([1,8,3],[1,8,2]),([2,8,4,1],[2,8,4,1]),([1,4,6,2],[1,8,2,1]):
            y=L(x,sel,rw,cfg_gu=cg,cfg_dn=cd).clone();rel=((y-r).norm()/r.norm()).item();worst=max(worst,rel)
            print(f'B{B} {mode:8s} {cg} {cd} rel {rel:.2e}',flush=True)
# lr must matter: dropping it must change the output well beyond the kernel error
for e in range(NE):ex[e].lr=None;L.set(e,ex[e],levels[e])
y0=L(x,sel,rw).clone();d=((y0-r).norm()/r.norm()).item()
print(f'worst {worst:.2e}; without lr {d:.2e}');print('LR PASS' if worst<2e-3 and d>20*worst else 'LR FAIL')
