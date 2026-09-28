# compute-bound probe: all table rows alias expert 0's planes (L2-hot) vs distinct experts (cold)
import torch;torch.cuda.set_per_process_memory_fraction(12/80)
from moe import *;from common import *;from timing import bench
H,I=6144,2048;NP=72;R=8;n3g,n3d=H//128//2,I//128//2
ex=[Expert(H,I,n3g,n3d,seed=i) for i in range(NP)]
res={}
for lv in [2,4]:
    Lc=MoELayer(NP,H,I,n3g,n3d);[Lc.set(e,ex[e],lv) for e in range(NP)]
    Lh=MoELayer(NP,H,I,n3g,n3d);[Lh.set(e,ex[0],lv) for e in range(NP)]
    for B in [1,4]:
        rts=routings(B,NP,R,mode='distinct');x=(torch.randn(B,H,device='cuda')*0.05).half()
        for part,which,c in [('gu',1,[1,8,6]),('dn',2,[1,4,4])]:
            fns={n:(lambda L=L:[L(x,s,w,which=which,cfg_gu=c,cfg_dn=c) for s,w in rts]) for n,L in [('cold',Lc),('hot',Lh)]}
            med,_=bench(fns,blocks=10,repeats=1)
            print(f'L{lv} B{B} {part}',{k:round(v/R,1) for k,v in med.items()},flush=True)
