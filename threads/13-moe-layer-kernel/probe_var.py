# diagnostic variants in one process, interleaved (hot planes, B1, 8 experts)
import torch;torch.cuda.set_per_process_memory_fraction(12/80)
from moe import *;from timing import bench,warmup;from build import get
H,I=6144,2048;NP=16;n3g,n3d=H//128//2,I//128//2
mods={'def':get([]),'fakehash':get(['NQ_FAKE_HASH']),'fakeload':get(['NQ_FAKE_LOAD']),'skip':get(['NQ_SKIP_GEMV'])}
ex=Expert(H,I,n3g,n3d,seed=0)
x=(torch.randn(1,H,device='cuda')*0.05).half();sel=torch.arange(8,device='cuda').view(1,8);rw=torch.full((1,8),1/8,device='cuda').half()
fns={}
for lv in [2,4]:
    for mn,m in mods.items():
        L=MoELayer(NP,H,I,n3g,n3d,mod=m);[L.set(e,ex,lv) for e in range(NP)]
        for part,which,c in [('gu',1,[1,8,8]),('dn',2,[1,4,4])]:
            fns[(lv,mn,part)]=(lambda L=L,which=which,c=c:L(x,sel,rw,which=which,cfg_gu=c,cfg_dn=c))
warmup()
med,_=bench(fns,blocks=20,repeats=8)
for k,v in med.items():print(k,round(v,1))
