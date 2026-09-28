"""NestQuant grouped MoE layer: pools, device table, launch wrapper, dense reference decode.
Plane layout: see nqmoe.cu header."""
import torch,numpy as np
from build import get
M=get()
TBL_W=16
A_=float(np.array([0x1eee],np.uint16).view(np.float16)[0]);B_=float(np.array([0xc931],np.uint16).view(np.float16)[0])

def proj_sizes(N,K,nm=None):
    """int32 words per plane (flags: int64 per strip). nm=None: dense P4; else nm flagged chunks per strip."""
    S,C=N//16,K//128;R=C if nm is None else nm
    return dict(S=S,C=C,nm=nm,base=S*C*32*4,p4=S*R*32*4,d4=S*R,flags=0 if nm is None else S)

def make_flags(S,C,nm,gen,grp=8):
    """nm flagged chunks per strip; identical across each group of `grp` strips (128x128 block granularity)."""
    ng=(S+grp-1)//grp
    idx=torch.rand(ng,C,generator=gen).argsort(1)[:,:nm]
    fl=torch.zeros(ng,C,dtype=torch.bool);fl.scatter_(1,idx,True)
    fl=fl.repeat_interleave(grp,0)[:S]
    return (fl.long()<<torch.arange(C)).sum(1),fl        # C <= 63

class Proj:
    """Random packed planes of one projection of one expert (separate tensors so they can live in pools)."""
    def __init__(s,N,K,gen,nm=None):
        z=proj_sizes(N,K,nm);s.z=z;s.N,s.K=N,K
        ri=lambda n:torch.randint(-2**31,2**31-1,(n,),generator=gen,dtype=torch.int32)
        dl=lambda n:(torch.rand(n,generator=gen)*0.2+0.2).half().repeat_interleave(2).view(torch.int32)
        s.base=ri(z['base']);s.p4=ri(z['p4']);s.d4=dl(z['d4'])
        if nm is None:s.flags=None;s.fl=None
        else:s.flags,s.fl=make_flags(z['S'],z['C'],nm,gen)
    def to(s,dev):
        for k in ['base','p4','d4','flags']:
            if getattr(s,k) is not None:setattr(s,k,getattr(s,k).to(dev))
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
    """Independent dense decode of one projection at level 2 or 4 -> [N,K] fp32 (EMU: fp16 rounding per HFMA2)."""
    z=p.z;S,C=z['S'],z['C'];dev=p.base.device
    u=lambda t:(t.long()&0xFFFFFFFF)
    hb=_codes(u(p.base).view(S,C,32,4),G);f=r16(A_*hb+B_)
    if level==4:
        R=C if p.fl is None else z['nm']
        h4=_codes(u(p.p4).view(S,R,32,4),G)
        d4=p.d4.view(torch.int16).view(torch.float16)[::2].double().view(S,R)
        if p.fl is None:on=torch.ones(S,C,dtype=torch.bool,device=dev);rank=torch.arange(C,device=dev)[None].expand(S,C)
        else:on=p.fl.to(dev);rank=(torch.cumsum(on.long(),1)-on.long()).clamp(max=R-1)
        si=torch.arange(S,device=dev)[:,None].expand(S,C)
        d=d4[si,rank][...,None,None];h=h4[si,rank]
        r4=r16(h*r16(d*A_)+r16(A_*hb+r16(d*B_+B_))) if EMU else d*(A_*h+B_)+f
        f=torch.where(on[...,None,None],r4,f)
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
    def __init__(s,H,I,seed,nm_gu=None,nm_dn=None,dev='cuda'):
        gen=torch.Generator().manual_seed(seed)
        s.gu=Proj(2*I,H,gen,nm_gu).to(dev);s.dn=Proj(H,I,gen,nm_dn).to(dev)
        s.signs=((torch.randint(0,2,(2*H+3*I,),generator=gen)*2-1).half()).to(dev)
        s.H,s.I=H,I
    def bytes(s,level):
        b=0
        for p in (s.gu,s.dn):
            b+=p.base.numel()*4
            if level>=4:b+=(p.p4.numel()+p.d4.numel())*4+(0 if p.flags is None else p.flags.numel()*8)
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
    for off,p in ((1,ex.gu),(5,ex.dn)):
        for i,k in enumerate(['base','p4','d4','flags']):
            t=getattr(p,k);e[off+i]=0 if t is None else t.data_ptr()
    e[9]=ex.signs.data_ptr()
    return e

class MoELayer:
    """Device table [E,16] + workspace for up to Bmax tokens. Table entries are flipped in place (graph-safe)."""
    def __init__(s,E,H,I,nm_gu=0,nm_dn=0,Bmax=4,topk=8,G=2,dev='cuda',mod=None):
        """nm_*: flagged chunks per strip for experts in mask mode (ignored for dense experts)."""
        s.M=mod or M;s.E,s.H,s.I,s.nm_gu,s.nm_dn,s.G=E,H,I,nm_gu,nm_dn,G
        s.table=torch.zeros(E,TBL_W,dtype=torch.int64,device=dev)
        S=Bmax*topk
        s.acc_gu=torch.zeros(S,2*I,device=dev);s.h=torch.zeros(S,I,device=dev).half();s.acc_d=torch.zeros(S,H,device=dev)
        s.cnt_gu=torch.zeros(S*(I//128),dtype=torch.int32,device=dev);s.cnt_d=torch.zeros(H//128,dtype=torch.int32,device=dev);s.wq=torch.zeros(4,dtype=torch.int32,device=dev)
        s.out=torch.zeros(Bmax,H,device=dev)
        s.cfg_gu=[1,8,3];s.cfg_dn=[1,8,2];s.hits_ptr=0   # set to a (host-mapped) int32 [E] pointer to export routing hits
    def set(s,e,ex,level):s.table[e].copy_(entry(ex,level).to(s.table.device),non_blocking=False)
    def __call__(s,x,sel,rw,out=None,force_level=0,which=3,cfg_gu=None,cfg_dn=None):
        out=s.out[:x.shape[0]] if out is None else out
        s.M.moe_forward(x,sel,rw,s.table,out,s.acc_gu,s.h,s.acc_d,s.cnt_gu,s.cnt_d,s.wq,s.I,s.nm_gu,s.nm_dn,
                      cfg_gu or s.cfg_gu,cfg_dn or s.cfg_dn,s.G,force_level,which,s.hits_ptr)
        return out

class Mailbox:
    """Graph-safe table updates: call .apply() inside the captured graph before the layer; stage from a side stream."""
    def __init__(s,layer):
        E=layer.E;dev=layer.table.device;s.L=layer
        s.stage=torch.zeros(E,TBL_W,dtype=torch.int64,device=dev);s.seq=torch.zeros(E,dtype=torch.int32,device=dev)
        s.applied=torch.zeros(E,dtype=torch.int32,device=dev);s.applied_host=torch.zeros(E,dtype=torch.int32).pin_memory()
        s.hseq=[0]*E;s.pins=[]
    def apply(s):s.L.M.mailbox(s.L.table,s.stage,s.seq,s.applied,s.applied_host.data_ptr())
    def post(s,e,row,stream):
        """enqueue on `stream` (after any P4 copy already enqueued there): stage[e]=row; seq[e]+=1"""
        assert s.done(e),'one outstanding op per expert'
        s.hseq[e]+=1;r=row.pin_memory();q=torch.tensor([s.hseq[e]],dtype=torch.int32).pin_memory();s.pins+=[r,q]
        with torch.cuda.stream(stream):s.stage[e].copy_(r,non_blocking=True);s.seq[e:e+1].copy_(q,non_blocking=True)
    def done(s,e):return int(s.applied_host[e])==s.hseq[e]

def moe_ref(experts,levels,x,sel,rw):
    y=torch.zeros(x.shape[0],x.shape[1],device=x.device);xf=x.float();cache={}
    for b in range(x.shape[0]):
        for k in range(sel.shape[1]):
            e=int(sel[b,k]);w=float(rw[b,k])
            if levels[e]<=0 or w==0:continue
            y[b]+=w*experts[e].ref(xf[b:b+1],levels[e])[0]
    return y
