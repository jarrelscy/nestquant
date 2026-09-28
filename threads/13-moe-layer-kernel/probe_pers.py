# grid-mapped vs persistent (work-queue) kernels, cold L2, per stage
import torch,json,itertools;torch.cuda.set_per_process_memory_fraction(12/80)
from moe import *;from common import *;from timing import bench,warmup
H,I=6144,2048;NP=72;R=8
ex=[Expert(H,I,seed=i) for i in range(NP)]
Ls={}
for lv in [2,4]:
    L=MoELayer(NP,H,I);[L.set(e,ex[e],lv) for e in range(NP)];Ls[lv]=L
tune=json.load(open('tune_I2048.json'))
warmup()
for B in [1,4]:
    rts=routings(B,NP,R);x=(torch.randn(B,H,device='cuda')*0.05).half()
    for part,which in [('gu',1),('dn',2)]:
        base=tune[f'{part}|B{B}']['cfg']
        cs=[tuple(base)]+[(c,sb,n,p) for c,sb,n,p in itertools.product([1,2],[8,4],[1,2,4] if part=='dn' else [2,3,4,6],[1,2]) if (H if part=='gu' else I)%(c*n*128)==0]
        fns={(lv,c):(lambda c=c,L=Ls[lv]:[L(x,s,w,which=which,cfg_gu=list(c),cfg_dn=list(c)) for s,w in rts]) for lv in [2,4] for c in cs}
        med,_=bench(fns,blocks=8,repeats=1)
        for lv in [2,4]:
            srt=sorted(cs,key=lambda c:med[(lv,c)])
            print(B,part,'L',lv,'grid',tuple(base),round(med[(lv,tuple(base))]/R,1),'| best',[(c,round(med[(lv,c)]/R,1)) for c in srt[:4]],flush=True)
