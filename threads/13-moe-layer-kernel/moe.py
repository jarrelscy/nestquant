"""NestQuant grouped MoE layer: pools, device table, launch wrapper, dense reference decode.
Plane layout: see nqmoe.cu header."""
import torch,numpy as np
from build import get
M=get()
TBL_W=16
A_=float(np.array([0x1eee],np.uint16).view(np.float16)[0]);B_=float(np.array([0xc931],np.uint16).view(np.float16)[0])

def proj_sizes(N,K,n3):
    S,C=N//16,K//128
    return dict(S=S,C=C,n3=n3,base=S*C*32*4,p3=S*n3*32*4,p4=S*(C-n3)*32*4,d3=S*n3,d4=S*(C-n3),flags=S)   # int32 words (flags: int64)

def make_flags(S,C,n3,gen,grp=8):
    """n3 flagged chunks per strip; identical across each group of `grp` strips (128x128 block granularity, thread 04)."""
    ng=(S+grp-1)//grp
    sc=torch.rand(ng,C,generator=gen)
    idx=sc.argsort(1)[:,:n3]
    fl=torch.zeros(ng,C,dtype=torch.bool);fl.scatter_(1,idx,True)
    fl=fl.repeat_interleave(grp,0)[:S]
    w=(fl.long()<<torch.arange(C)).sum(1)            # C<=63 fits in int64 without sign issues for C<=63
    return w,fl

class Proj:
    """Random packed planes of one projection of one expert (separate tensors so they can live in pools)."""
    def __init__(s,N,K,n3,gen):
        z=proj_sizes(N,K,n3);s.z=z;s.N,s.K=N,K
        ri=lambda n:torch.randint(-2**31,2**31-1,(n,),generator=gen,dtype=torch.int32)
        dl=lambda n:(torch.rand(n,generator=gen)*0.2+0.2).half().repeat_interleave(2).view(torch.int32)
        s.base=ri(z['base']);s.p3=ri(z['p3']);s.p4=ri(z['p4']);s.d3=dl(z['d3']);s.d4=dl(z['d4'])
        w,s.fl=make_flags(z['S'],z['C'],n3,gen);s.flags=w
    def to(s,dev):
        for k in ['base','p3','p4','d3','d4','flags']:setattr(s,k,getattr(s,k).to(dev))
        return s

def _codes(rec,G=2):
    """rec [S,C,32,4] int64 (uint32 values) -> [S,C,32,64] fp64 codes 1024+bytesum (2 bit/weight windows)."""
    lane=torch.arange(32,device=rec.device);src=lane^1 if G==2 else ((lane&~3)|((lane+1)&3))
    ext=torch.cat([rec,rec[:,:,src,:1]],-1)
    out=[]
    for j in range(64):
        o=2*j;i,sh=o>>5,o&31
        v=((ext[...,i]>>sh)|(ext[...,i+1]<<(32-sh))) if sh else ext[...,i]
        v=v&0xFFFF;x=(v*0x83DCD12D)&0xFFFFFFFF
        out.append(1024+((x&255)+((x>>8)&255)+((x>>16)&255)+((x>>24)&255)))
    return torch.stack(out,-1).double()

EMU=True   # emulate the kernel's fp16 HFMA2 rounding (delta folded into B*(1+delta), delta*A)
def r16(t):return t.half().double() if EMU else t
def dense_W(p,level,G=2):
    """Independent dense decode of one projection at a level -> [N,K] fp32. EMU: fp16 single-rounding per HFMA2."""
    z=p.z;S,C,n3=z['S'],z['C'],z['n3'];dev=p.base.device
    u=lambda t:(t.long()&0xFFFFFFFF)
    hb=_codes(u(p.base).view(S,C,32,4),G);f=r16(A_*hb+B_)
    if level>=3:
        fl=p.fl.to(dev)                                         # [S,C] bool
        rank=torch.cumsum(fl.long(),1)-fl.long()               # rank among flagged
        rank4=torch.arange(C,device=dev)[None]-rank            # rank among unflagged
        h3=_codes(u(p.p3).view(S,n3,32,4),G) if n3 else None
        h4=_codes(u(p.p4).view(S,C-n3,32,4),G) if C-n3 else None
        d3=p.d3.view(torch.int16).view(torch.float16)[::2].double().view(S,n3) if n3 else None
        d4=p.d4.view(torch.int16).view(torch.float16)[::2].double().view(S,C-n3) if C-n3 else None
        si=torch.arange(S,device=dev)[:,None].expand(S,C)
        r=f.clone()
        if n3:
            m=fl;rr=rank.clamp(max=n3-1)
            d=d3[si,rr][...,None,None]
            r3=r16(h3[si,rr]*r16(d*A_)+r16(A_*hb+r16(d*B_+B_))) if EMU else d*(A_*h3[si,rr]+B_)+f
            r=torch.where(m[...,None,None],r3,r)
        if level==4 and C-n3:
            m=~fl;rr=rank4.clamp(max=C-n3-1)
            d=d4[si,rr][...,None,None]
            r4=r16(h4[si,rr]*r16(d*A_)+r16(A_*hb+r16(d*B_+B_))) if EMU else d*(A_*h4[si,rr]+B_)+f
            r=torch.where(m[...,None,None],r4,r)
        on=fl if level==3 else torch.ones_like(fl)
        f=torch.where(on[...,None,None],r,f)
    W=torch.zeros(p.N,p.K,device=dev,dtype=torch.float64)
    lane=torch.arange(32,device=dev);g=lane>>2;t4=lane&3
    sI=torch.arange(S,device=dev)[:,None,None];cI=torch.arange(C,device=dev)[None,:,None]
    for j in range(64):
        pp,e=j>>1,j&1;t,r_=pp>>2,pp&3
        rows=(sI*16+g[None,None]+(r_&1)*8).expand(S,C,32);ks=(cI*128+t*16+t4[None,None]*2+(r_>>1)*8+e).expand(S,C,32)
        W[rows,ks]=f[...,j]
    return W.float()

def H128(dev='cuda'):
    H=torch.ones(1,1)
    while H.shape[0]<128:H=torch.cat([torch.cat([H,H],1),torch.cat([H,-H],1)],0)
    return (H/128**0.5).to(dev)

class Expert:
    def __init__(s,H,I,n3_gu,n3_dn,seed,dev='cuda'):
        gen=torch.Generator().manual_seed(seed)
        s.gu=Proj(2*I,H,n3_gu,gen).to(dev);s.dn=Proj(H,I,n3_dn,gen).to(dev)
        s.signs=((torch.randint(0,2,(2*H+3*I,),generator=gen)*2-1).half()).to(dev)
        s.H,s.I=H,I
    def bytes(s,level):
        b=0
        for p in (s.gu,s.dn):
            b+=p.base.numel()*4+p.flags.numel()*8
            if level>=3:b+=(p.p3.numel()+p.d3.numel())*4
            if level>=4:b+=(p.p4.numel()+p.d4.numel())*4
        return b
    def ref(s,x,level):
        """x [T,H] fp32 -> [T,H] fp32, through the dense-decoded weights."""
        H,I=s.H,s.I;Hm=H128(x.device);wht=lambda v:(v.view(*v.shape[:-1],-1,128)@Hm).view(v.shape)
        sg=s.signs.float();su,svg,svu,sud,svo=sg[:H],sg[H:H+I],sg[H+I:H+2*I],sg[H+2*I:H+3*I],sg[H+3*I:]
        Wg=dense_W(s.gu,level);Wd=dense_W(s.dn,level)
        xr=wht(x*su).half().float();a=xr@Wg.T
        g=wht(a[:,:I])*svg;u=wht(a[:,I:])*svu
        h=wht(torch.nn.functional.silu(g)*u*sud).half().float()
        return wht(h@Wd.T)*svo

def entry(ex,level):
    e=torch.zeros(TBL_W,dtype=torch.int64)
    e[0]=level
    for off,p in ((1,ex.gu),(7,ex.dn)):
        for i,k in enumerate(['base','p3','p4','d3','d4','flags']):e[off+i]=getattr(p,k).data_ptr()
    e[13]=ex.signs.data_ptr()
    return e

class MoELayer:
    """Device table [E,16] + workspace for up to Bmax tokens. Table entries are flipped in place (graph-safe)."""
    def __init__(s,E,H,I,n3_gu,n3_dn,Bmax=4,topk=8,G=2,dev='cuda'):
        s.E,s.H,s.I,s.n3_gu,s.n3_dn,s.G=E,H,I,n3_gu,n3_dn,G
        s.table=torch.zeros(E,TBL_W,dtype=torch.int64,device=dev)
        S=Bmax*topk
        s.acc_gu=torch.zeros(S,2*I,device=dev);s.h=torch.zeros(S,I,device=dev).half();s.acc_d=torch.zeros(S,H,device=dev)
        s.cnt_gu=torch.zeros(S*(I//128),dtype=torch.int32,device=dev);s.cnt_d=torch.zeros(H//128,dtype=torch.int32,device=dev)
        s.out=torch.zeros(Bmax,H,device=dev)
        s.cfg_gu=[1,8,3];s.cfg_dn=[1,8,2]
    def set(s,e,ex,level):s.table[e].copy_(entry(ex,level).to(s.table.device),non_blocking=False)
    def __call__(s,x,sel,rw,out=None,force_level=0,which=3,cfg_gu=None,cfg_dn=None):
        out=s.out[:x.shape[0]] if out is None else out
        M.moe_forward(x,sel,rw,s.table,out,s.acc_gu,s.h,s.acc_d,s.cnt_gu,s.cnt_d,s.I,s.n3_gu,s.n3_dn,
                      cfg_gu or s.cfg_gu,cfg_dn or s.cfg_dn,s.G,force_level,which)
        return out

def moe_ref(experts,levels,x,sel,rw):
    y=torch.zeros(x.shape[0],x.shape[1],device=x.device);xf=x.float();cache={}
    for b in range(x.shape[0]):
        for k in range(sel.shape[1]):
            e=int(sel[b,k]);w=float(rw[b,k])
            if levels[e]<=0 or w==0:continue
            y[b]+=w*experts[e].ref(xf[b:b+1],levels[e])[0]
    return y
