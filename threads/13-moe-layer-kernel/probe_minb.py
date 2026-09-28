import torch,sys;torch.cuda.set_per_process_memory_fraction(12/80)
from moe import *;from timing import bench,warmup;from build import get
H,I=6144,2048;NP=16;n3g,n3d=H//128//2,I//128//2
mods={f'minb{b}':get([f'NQ_MINB={b}'] if b>1 else []) for b in [1,3,4]}
for n,m in mods.items():print(n,[m.occ(md,2,c,8,16384) for md in [0,1] for c in [1,2]])
ex=Expert(H,I,n3g,n3d,seed=0)
x=(torch.randn(1,H,device='cuda')*0.05).half();sel=torch.arange(8,device='cuda').view(1,8);rw=torch.full((1,8),1/8,device='cuda').half()
fns={}
for lv in [2,4]:
    for mn,m in mods.items():
        L=MoELayer(NP,H,I,n3g,n3d,mod=m);[L.set(e,ex,lv) for e in range(NP)]
        for part,which in [('gu',1),('dn',2)]:
            for c in ([[1,8,8],[1,8,4],[2,8,4]] if part=='gu' else [[1,8,4],[2,8,2],[1,8,2]]):
                fns[(lv,mn,part,str(c))]=(lambda L=L,which=which,c=c:L(x,sel,rw,which=which,cfg_gu=c,cfg_dn=c))
warmup()
med,_=bench(fns,blocks=10,repeats=8)
for k,v in med.items():print(k,round(v,1))
