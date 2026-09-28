import torch,itertools
from build import get
M=get()
DEC=dict(UNI2=0,UNI4=1,T2=2,T4=3,H2=4,H4=5,T2R1=6,T2R2=7,T2H=8,SYN4=9,SYN2=10,H2B=11,J4=12,J3=13)
NB={'UNI4':8,'T4':8,'H4':8,'SYN4':8}
PLANES={'J4':(1,1),'J3':(1,0),'T2R1':(1,0),'T2R2':(1,1),'T2H':(1,1)}
LUTD={'H2','H4','T2H','H2B'}
E=torch.empty(0,device='cuda')
class Proj:
    """Random packed weights for one [N,K] projection in decode format `dec`."""
    def __init__(s,dec,N,K,split=True,lutr=False,r=0):
        s.dec,s.N,s.K,s.split,s.lutr,s.r=dec,N,K,split,lutr,r
        strips,chunks=N//16,K//128;n=strips*chunks*32
        nb=NB.get(dec,4);p3,p4=PLANES.get(dec,(0,0))
        ri=lambda *sh:torch.randint(-2**31,2**31-1,sh,device='cuda',dtype=torch.int32)
        if split or not p3:
            s.base=ri(n*nb);s.p3=ri(n*2) if p3 else E;s.p4=ri(n*2) if p4 else E;s.istride=nb//4
        else:
            s.base=ri(n*8);s.p3=E;s.p4=E;s.istride=2
        s.lut=(torch.randn(512*2,device='cuda')*0.5).half().view(torch.int32) if dec in LUTD else E
        s.bytes=s.base.numel()*4+s.p3.numel()*4+s.p4.numel()*4
        s.cfg=(1,4,2,1)
    def __call__(s,x,y,cfg=None,wdbg=E):
        cpw,sb,wk,nst=cfg or s.cfg
        M.gemv(x,s.base,s.p3,s.p4,s.lut,y,DEC[s.dec],s.lutr,s.split,s.r,cpw,sb,wk,s.N,s.K,wdbg,s.istride,nst)
    def configs(s):
        out=[]
        for cpw,sb,wk,nst in itertools.product([1,2],[2,4,8],[1,2,4],[1,2,3,4,6,8]):
            if sb*wk not in (4,8):continue
            if (s.N//16)%sb or s.K%(wk*cpw*nst*128):continue
            if wk*cpw*nst*128*4*2>40000:continue
            out.append((cpw,sb,wk,nst))
        return out
def check(p,B=3):
    x=torch.randn(B,p.K,device='cuda').half();y=torch.zeros(B,p.N,device='cuda');W=torch.zeros(p.N,p.K,device='cuda')
    p(x,y,wdbg=W);y2=torch.zeros(B,p.N,device='cuda');p(x,y2)
    ref=x.float()@W.T
    return ((y2-ref).norm()/ref.norm()).item(), W.std().item(), (W==0).float().mean().item()
