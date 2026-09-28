# variants in one process, interleaved; hot planes (all rows alias one expert), B1 and B4-distinct
import torch,sys,os;torch.cuda.set_per_process_memory_fraction(12/80)
from moe import *;from timing import bench,warmup;from build import get
H,I=6144,2048;NP=40
V=[v.split('+') if v!='def' else [] for v in sys.argv[1].split(',')]
mods={'+'.join(d) or 'def':get(d) for d in V}
for n,m in mods.items():print(n,[m.occ(md,2,c,8,16384) for md in [0,1] for c in [1,2]])
ex=Expert(H,I,seed=0)
cf={'gu':[[1,8,8],[2,8,4],[1,8,6]],'dn':[[1,8,4],[2,8,2],[1,8,8]]}
fns={}
for B in [1,4]:
    x=(torch.randn(B,H,device='cuda')*0.05).half();sel=torch.arange(8*B,device='cuda').view(B,8);rw=torch.full((B,8),1/8,device='cuda').half()
    for lv in [2,4]:
        for mn,m in mods.items():
            L=MoELayer(NP,H,I,mod=m);[L.set(e,ex,lv) for e in range(NP)]
            for part,which in [('gu',1),('dn',2)]:
                for c in cf[part]:
                    fns[(B,lv,mn,part,str(c))]=(lambda L=L,which=which,c=c,x=x,sel=sel,rw=rw:L(x,sel,rw,which=which,cfg_gu=c,cfg_dn=c))
warmup()
med,_=bench(fns,blocks=10,repeats=4)
best={}
for k,v in med.items():
    kk=k[:4];best[kk]=min(best.get(kk,(1e9,)),(round(v,1),k[4]))
for k,v in best.items():print(k,v)
