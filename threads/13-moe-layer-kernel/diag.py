import torch;torch.cuda.set_per_process_memory_fraction(12/80)
from moe import *
H,I=6144,2048
e=Expert(H,I,24,8,seed=5);L=MoELayer(1,H,I,24,8)
for lv in [2,3,4]:
    L.set(0,e,lv)
    x=(torch.randn(1,H,device='cuda')*0.05).half();sel=torch.zeros(1,8,dtype=torch.long,device='cuda');sel[0,1:]=-1
    rw=torch.zeros(1,8,device='cuda').half();rw[0,0]=1
    sel[0,1:]=0  # duplicates within a token are not legal; use weights 0 instead
    L(x,sel,rw,which=1);h=L.h[0].float().clone()
    # reference h
    Hm=H128();wht=lambda v:(v.view(*v.shape[:-1],-1,128)@Hm).view(v.shape)
    sg=e.signs.float();su,svg,svu,sud=sg[:H],sg[H:H+I],sg[H+I:H+2*I],sg[H+2*I:H+3*I]
    Wg=dense_W(e.gu,lv);xr=wht(x.float()*su).half().float();a=xr@Wg.T
    g=wht(a[:,:I])*svg;u=wht(a[:,I:])*svu;hr=wht(torch.nn.functional.silu(g)*u*sud)[0]
    print(lv,'h rel',((h-hr).norm()/hr.norm()).item(),'a norm',a.norm().item(), 'W std',Wg.std().item(),'W mean',Wg.mean().item())
    L.acc_gu.zero_();L.cnt_gu.zero_()
