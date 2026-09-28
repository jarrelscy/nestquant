"""v2 projections: additive progressive trellis, shared-lane rings, optional fusion. Random packed data, real shapes."""
import torch,itertools,numpy as np
from build2 import get
M=get()
DEC=dict(B2=0,T4=1,A3=2,A4=3,MIX=4,T2H=5)
E=torch.empty(0,device='cuda')
class Proj2:
    def __init__(s,dec,N,K,G=2,mixfrac=0.5,seed=None,strip=False,grp=1):
        s.dec,s.N,s.K,s.G=dec,N,K,G
        S,C=N//16,K//128;n=S*C*32
        ri=lambda *sh:torch.randint(-2**31,2**31-1,sh,device='cuda',dtype=torch.int32)
        s.base=ri(n*(8 if dec=='T4' else 4))
        s.p3=ri(n*2) if dec=='A3' else E
        s.p4=ri(n*4) if dec in('A4','MIX','T2H') else E
        s.lut=(torch.randn(1024,device='cuda')*0.5).half().view(torch.int32) if dec=='T2H' else E
        s.delta=(torch.rand(S*C,device='cuda')*0.2+0.2).half().repeat_interleave(2).view(torch.int32) if dec in('A3','A4','MIX','T2H') else E
        if dec=='MIX':
            fl=(torch.rand(S,(C+31)//32*32,device='cuda')<mixfrac)
            if strip:fl[:]=fl[:,:1]
            if grp>1:fl=fl[::grp].repeat_interleave(grp,0)
            fl[:,C:]=False
            w=(fl.view(S,-1,32).long()<<torch.arange(32,device='cuda')).sum(-1)
            s.flags=w.to(torch.int64).where(w<2**31,w-2**32).to(torch.int32).contiguous();s.fl=fl[:,:C]
        else:s.flags=E
        nb={'B2':4,'T4':8,'A3':6,'A4':8}.get(dec)
        s.bytes=(s.base.numel()+s.p3.numel()+(s.p4.numel()*(mixfrac if dec=='MIX' else 1)))*4+s.delta.numel()*4
        s.cfg=(1,8,3)
    def __call__(s,x,acc,cfg=None,mode=0,ex=(),wdbg=E,dbg=0):
        cpw,sb,nst=cfg or s.cfg
        M.gemv2(x,s.base,s.p3,s.p4,s.delta,s.flags,s.lut,acc,DEC[s.dec],s.G,cpw,sb,nst,s.N,s.K,mode,wdbg,list(ex),dbg)
    def configs(s,fused=False):
        out=[]
        for cpw,sb,nst in itertools.product([1,2],[4,8],[1,2,3,4,6,8]):
            if (s.N//16)%sb or s.K%(cpw*nst*128):continue
            if cpw*nst*128*4*2>40000:continue
            out.append((cpw,sb,nst))
        return out
A_=float(np.array([0x1eee],np.uint16).view(np.float16)[0]);B_=float(np.array([0xc931],np.uint16).view(np.float16)[0])
def _codes(rec,G,bpw):
    """rec: [S,C,32,W] uint32 own records. returns h codes [S,C,32,64] (1024+bytesum) in lane-weight order (2p+e)."""
    S,C,L,W=rec.shape
    lane=np.arange(32);src=lane if G==1 else (lane^1 if G==2 else (lane&~3)|((lane+1)&3))
    ext=np.concatenate([rec,rec[:,:,src,:1]],-1).astype(np.uint64)
    out=np.zeros((S,C,L,64),np.float64)
    for j in range(64):
        o=bpw*j;i,sh=o>>5,o&31
        v=(ext[...,i]>>sh)|(ext[...,i+1]<<(32-sh)) if sh else ext[...,i]
        v=v&0xFFFF;x=(v*0x83DCD12D)&0xFFFFFFFF
        out[...,j]=1024+sum((x>>(8*b))&0xFF for b in range(4))
    return out
def ref_W(p):
    S,C=p.N//16,p.K//128
    rec=lambda t,w:t.cpu().numpy().view(np.uint32).reshape(S,C,32,w)
    if p.dec=='T4':f=A_*_codes(rec(p.base,8),p.G,4)+B_
    else:
        f=A_*_codes(rec(p.base,4),p.G,2)+B_
        if p.dec!='B2':
            d=p.delta.cpu().view(torch.int16).numpy().view(np.float16)[::2].astype(np.float64).reshape(S,C,1,1)
            hr=_codes(rec(p.p3,2),p.G,1) if p.dec=='A3' else _codes(rec(p.p4,4),p.G,2)
            r=d*(A_*hr+B_)
            if p.dec=='T2H':
                lut=p.lut.cpu().view(torch.int16).numpy().view(np.float16).astype(np.float64)
                rr=rec(p.p4,4);lane=np.arange(32);src=lane if p.G==1 else (lane^1 if p.G==2 else (lane&~3)|((lane+1)&3))
                ext=np.concatenate([rr,rr[:,:,src,:1]],-1).astype(np.uint64);r=np.zeros_like(f)
                for pp in range(32):
                    o=4*pp;i,sh=o>>5,o&31
                    v=((ext[...,i]>>sh)|(ext[...,i+1]<<(32-sh)) if sh else ext[...,i])&0xFFFF
                    idx=(((v*v+v)&0xFFFFFFFF)>>7)&0x1FF
                    r[...,2*pp]=d[...,0]*lut[2*idx];r[...,2*pp+1]=d[...,0]*lut[2*idx+1]
            if p.dec=='MIX':r=r*p.fl.cpu().numpy()[:,:,None,None]
            f=f+r
    W=np.zeros((p.N,p.K))
    for j in range(64):
        pp,e=j>>1,j&1;t,r=pp>>2,pp&3
        g=np.arange(32)>>2;t4=np.arange(32)&3
        rows=(np.arange(S)[:,None,None]*16+g[None,None,:]+(r&1)*8)
        ks=(np.arange(C)[None,:,None]*128+t*16+t4[None,None,:]*2+(r>>1)*8+e)
        W[np.broadcast_to(rows,(S,C,32)),np.broadcast_to(ks,(S,C,32))]=f[...,j]
    return W
def check(p,B=3,cfg=None):
    x=torch.randn(B,p.K,device='cuda').half();y=torch.zeros(B,p.N,device='cuda');W=torch.zeros(p.N,p.K,device='cuda')
    p(x,y,cfg,wdbg=W);y2=torch.zeros(B,p.N,device='cuda');p(x,y2,cfg)
    ref=x.float()@W.T;Wr=torch.tensor(ref_W(p),device='cuda',dtype=torch.float32)
    return ((y2-ref).norm()/ref.norm()).item(),(W-Wr).abs().max().item(),W.std().item()
