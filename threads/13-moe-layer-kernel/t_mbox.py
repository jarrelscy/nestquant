import torch,json;torch.cuda.set_per_process_memory_fraction(12/80)
from moe import *;from common import *;from timing import bench,warmup
H,I=6144,256;NP=256;R=8
ex=[Expert(H,I,seed=i) for i in range(NP)];L=MoELayer(NP,H,I);[L.set(e,ex[e],2) for e in range(NP)];MB=Mailbox(L)
hits=torch.zeros(NP,dtype=torch.int32).pin_memory()
warmup();rts=routings(1,NP,R);x=(torch.randn(1,H,device='cuda')*0.05).half();L.cfg_gu=[1,4,6];L.cfg_dn=[2,8,1]
def run(mb,hp):
    def f():
        L.hits_ptr=hp
        for s,w in rts:
            if mb:MB.apply()
            L(x,s,w)
    return f
med,_=bench({'plain':run(0,0),'mailbox':run(1,0),'mailbox+hits':run(1,hits.data_ptr())},blocks=40,repeats=1)
print({k:round(v/R,2) for k,v in med.items()},'us/layer, I=256 B1 2b')
