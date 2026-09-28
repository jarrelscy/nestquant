# time vs number of active experts at B1 (hot planes): intercept = fixed cost, slope = per-expert cost
import torch;torch.cuda.set_per_process_memory_fraction(12/80)
from moe import *;from common import *;from timing import bench
H,I=6144,2048;NP=16;n3g,n3d=H//128//2,I//128//2
ex=[Expert(H,I,n3g,n3d,seed=i) for i in range(2)]
for lv in [2,4]:
    L=MoELayer(NP,H,I,n3g,n3d);[L.set(e,ex[0],lv) for e in range(NP)]
    x=(torch.randn(1,H,device='cuda')*0.05).half()
    sel=torch.arange(8,device='cuda').view(1,8)
    fns={}
    for n in [1,2,4,8]:
        rw=torch.zeros(1,8,device='cuda').half();rw[0,:n]=1/n
        for part,which,c in [('gu',1,[1,8,8]),('dn',2,[1,4,4])]:
            fns[(part,n)]=(lambda rw=rw,which=which,c=c:L(x,sel,rw,which=which,cfg_gu=c,cfg_dn=c))
    med,_=bench(fns,blocks=10,repeats=8)
    for k,v in med.items():print('L',lv,k,round(v,1))
