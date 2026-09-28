"""Low-rank plane (table [14]/[15]) vs the torch reference: synthetic experts with (r_gu, r_dn) in
{(0,0),(1,0),(0,1),(1,1),(2,1),(4,4)}, mixed levels 2/4 in one launch, both kernel variants (grid + persistent), G=4,
I = 2048 and 256 (TP8 shard). Also: workspace (z_gu half of zws, counters, accumulators) is zero after every call (z_dn partials are overwritten per call), repeat stability."""
import torch;torch.cuda.set_per_process_memory_fraction(12/80)
from moe import *;from common import routings
torch.manual_seed(0)
RR=[(0,0),(1,0),(0,1),(1,1),(2,1),(4,4)]
ok=True
for I in (2048,256):
    H=6144;NP=40
    ex=[Expert(H,I,seed=900+i,rk_gu=6 if i%2 else 0,rk_dn=7,var=bool(i%3)) for i in range(NP)]
    for i,e in enumerate(ex):
        rg,rd=RR[i%len(RR)]
        if rg+rd:e.rand_lr(rg,rd,seed=i,scale=0.04)
    lv=[4 if i%4<2 else 2 for i in range(NP)]
    L=MoELayer(NP,H,I);[L.set(i,ex[i],lv[i]) for i in range(NP)]
    for cfg in ([[1,8,3],[1,8,2]],[[2,8,4,1],[2,8,4,1]],[[1,4,6],[1,8,2]]):
        cg,cd=cfg
        if I==256:cg=[cg[0],cg[1],min(cg[2],2)]+cg[3:];cd=[cd[0],cd[1],1]+cd[3:]
        worst=0;worst0=1e9
        for B in (1,2,3,4):
            for s,w in routings(B,NP,8)[:3]:
                x=(torch.randn(B,H,device='cuda')*0.05).half()
                y=L(x,s,w,cfg_gu=cg,cfg_dn=cd).clone();y2=L(x,s,w,cfg_gu=cg,cfg_dn=cd).clone()
                r=moe_ref(ex,lv,x,s,w)
                # the same reference with the lr term dropped: shows the lr term is actually exercised
                keep=[e.lr for e in ex];
                for e in ex:e.lr=None
                r0=moe_ref(ex,lv,x,s,w)
                for e,k in zip(ex,keep):e.lr=k
                rel=((y-r).norm()/r.norm()).item();rel0=((y-r0).norm()/r0.norm()).item()
                worst=max(worst,rel,((y2-r).norm()/r.norm()).item());worst0=min(worst0,rel0)
        clean=L.zws[:32*(I//128)*4].abs().max().item()==0 and L.wq.abs().sum().item()==0 and L.acc_gu.abs().max().item()==0 and L.acc_d.abs().max().item()==0
        good=worst<3e-4 and clean and worst0>10*worst;ok&=good
        print(f'I {I} cfg {cg} {cd}: max rel err {worst:.2e} (vs no-lr ref >= {worst0:.2e}) ws clean {clean}',flush=True)
print('PASS' if ok else 'FAIL')
