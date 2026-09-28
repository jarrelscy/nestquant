"""Fused 2-launch chain (MODE1 gate/up + SwiGLU epilogue, MODE2 down + output WHT) vs torch reference on dense-decoded W."""
import torch;torch.cuda.set_per_process_memory_fraction(12/80)
from nq2 import *
def H128():
    H=torch.ones(1,1)
    while H.shape[0]<128:H=torch.cat([torch.cat([H,H],1),torch.cat([H,-H],1)],0)
    return H.cuda()/128**0.5
H=H128()
wht=lambda v:(v.view(*v.shape[:-1],-1,128)@H).view(v.shape)
sg=lambda n:(torch.randint(0,2,(n,),device='cuda')*2-1).half()
for d in __import__('os').environ.get('DECS','B2,A4,MIX,A3').split(','):
  for B in [1,4]:
    gu=Proj2(d,4096,6144,G=2);dn=Proj2(d,6144,2048,G=2)
    su_x,sv_g,sv_u,su_d,sv_o=sg(6144),sg(2048),sg(2048),sg(2048),sg(6144)
    x=torch.randn(B,6144,device='cuda')*0.01
    acc=torch.zeros(B,4096,device='cuda');hh=torch.empty(B,2048,device='cuda').half()
    y=torch.zeros(B,6144,device='cuda');out=torch.empty(B,6144,device='cuda').half()
    c1=torch.zeros(16,dtype=torch.int32,device='cuda');c2=torch.zeros(48,dtype=torch.int32,device='cuda')
    import os
    if os.environ.get('SPLIT'):
        accs=[torch.zeros(B,4096,device='cuda'),torch.zeros(B,4096,device='cuda')]
        for it in range(3):
            gu(x,accs[it%2],(1,8,3),mode=3,ex=(su_x,));dn(hh,y,(1,8,4),mode=4,ex=(accs[it%2],accs[(it+1)%2],sv_g,sv_u,su_d,sv_o,out,c2))
        acc=accs[0]+accs[1]
    else:
      for it in range(3):
        gu(x,acc,(1,8,3),mode=1,ex=(su_x,sv_g,sv_u,su_d,hh,c1));dn(hh,y,(1,8,4),mode=2,ex=(sv_o,out,c2))
    Wg=torch.zeros(4096,6144,device='cuda');Wd=torch.zeros(6144,2048,device='cuda')
    gu(torch.zeros(1,6144,device='cuda').half(),torch.zeros(1,4096,device='cuda'),wdbg=Wg,cfg=(1,8,3));dn(torch.zeros(1,2048,device='cuda').half(),torch.zeros(1,6144,device='cuda'),wdbg=Wd,cfg=(1,8,4))
    xh=wht(x*su_x.float()).half().float();a=xh@Wg.T
    g=wht(a[:,:2048])*sv_g.float();u=wht(a[:,2048:])*sv_u.float()
    h=wht(torch.nn.functional.silu(g)*u*su_d.float()).half().float()
    r=wht((h@Wd.T)*sv_o.float())
    print(d,B,'rel err',((out.float()-r).norm()/r.norm()).item(),'acc left',acc.abs().max().item(),y.abs().max().item(),c1.sum().item(),c2.sum().item())
