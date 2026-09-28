"""Thread 15 projections: generic base/residual trellis planes with fractional K, sub-array packing; numpy reference."""
import torch,itertools,numpy as np
from build15 import get
M=get()
E=torch.empty(0,device='cuda')
RM=dict(NONE=0,A=1,F=2,A2=3,F2=4,UNI4=5,P=6,P2=7)
def popc(m):return bin(m).count('1')
def step_off(j,KA,MASK):
    s=(j>>4)*(16*KA+popc(MASK))
    for i in range(j&15):s+=KA+((MASK>>i)&1)
    return s
def split_bits(bits):
    n4=bits//128;r=bits%128;return n4,(r//64),(r%64)//32,(r%32)//16
def pack(words,bits):
    """words [nrec, NW] uint32 -> flat uint8 sub-array layout (uint4 x n4 | uint2 | uint | ushort), each contiguous over records."""
    n4,n2,n1,nh=split_bits(bits);k=0;parts=[]
    parts.append(words[:,:4*n4].reshape(-1).view(np.uint8));k=4*n4
    if n2:parts.append(words[:,k:k+2].reshape(-1).view(np.uint8));k+=2
    if n1:parts.append(np.ascontiguousarray(words[:,k]).view(np.uint8));k+=1
    if nh:parts.append(np.ascontiguousarray(words[:,k].astype(np.uint16)).view(np.uint8))
    return np.concatenate(parts)
A_=float(np.array([0x1eee],np.uint16).view(np.float16)[0]);B_=float(np.array([0xc931],np.uint16).view(np.float16)[0])
class Proj:
    def __init__(s,vid,N,K,seed=0):
        s.vid,s.N,s.K=vid,N,K
        s.bbits,s.rbits,s.hasr,s.rmode,s.v2,s.bka,s.bm,s.rka,s.rm,s.wopt,s.raw=M.info(vid)
        S,C=N//16,K//128;s.nrec=n=S*C*32
        rng=np.random.default_rng(seed)
        def plane(bits):
            nw=(bits+31)//32;w=rng.integers(0,2**32,(n,nw),dtype=np.uint64).astype(np.uint32)
            if bits%32:w[:,-1]&=0xFFFF
            return w
        s.wb=plane(s.bbits);s.base=torch.tensor(pack(s.wb,s.bbits)).cuda()
        if s.hasr:
            s.wr=plane(s.rbits);s.res=torch.tensor(pack(s.wr,s.rbits)).cuda()
            if s.rmode in(RM['A'],RM['A2']):
                d=(rng.random(S*C)*0.2+0.2).astype(np.float16);s.dl=d.astype(np.float64)
                s.delta=torch.tensor(np.repeat(d,2).view(np.int32)).cuda()
            elif s.rmode in(RM['P'],RM['P2']):
                Nn=rng.integers(20,120 if s.v2 else 200,S*C).astype(np.int32);Mb=np.minimum(255,257-Nn)
                s.Mb,s.Nn=Mb,Nn;s.dl=Nn/Mb
                s.delta=torch.tensor((Mb|(Nn<<8)).astype(np.int32)).cuda()
            else:
                Nn=rng.integers(26,52,S*C).astype(np.int32);s.dl=Nn/128.
                s.delta=torch.tensor(Nn).cuda()
        else:s.res=E;s.delta=E
        s.bytes=s.base.numel()+s.res.numel()+(S*C*(2 if s.hasr else 0))   # delta stored as fp16/u8+pad per 16x128 block
        s.bpw=s.bytes*8/(N*K)
        s.cfg=(1,8,3)
    def __call__(s,x,acc,cfg=None,mode=0,ex=(),wdbg=E):
        cpw,sb,nst=cfg or s.cfg
        M.gemv(x,s.base,s.res,s.delta,acc,s.vid,cpw,sb,nst,s.N,s.K,mode,wdbg,list(ex))
    def configs(s):
        out=[]
        for cpw,sb,nst in itertools.product([1,2],[4,8],[1,2,3,4,6,8]):
            if (s.N//16)%sb or s.K%(cpw*nst*128):continue
            if cpw*nst*128*4*2>40000:continue
            out.append((cpw,sb,nst))
        return out
def _ext(w,bits):
    n=w.shape[0];src=(np.arange(n)^1)   # lane^1 within the warp (records are lane-minor)
    nb=w[src,0].astype(np.uint64);e=w.astype(np.uint64)
    if bits%32==0:e=np.concatenate([e,nb[:,None],np.zeros((n,1),np.uint64)],1)
    else:
        e[:,-1]=(e[:,-1]&0xFFFF)|((nb<<16)&0xFFFFFFFF);e=np.concatenate([e,(nb>>16)[:,None],np.zeros((n,1),np.uint64)],1)
    return e
def _win(e,o):
    i,sh=o>>5,o&31
    v=((e[:,i]>>sh)|(e[:,i+1]<<(32-sh))) if sh else e[:,i]
    return v&0xFFFF
def _bytes(v,mult=(1,1,1,1)):
    x=(v*0x83DCD12D)&0xFFFFFFFF
    return sum(m*((x>>(8*b))&0xFF).astype(np.int64) for b,m in enumerate(mult))
def ref_vals(p):
    """[nrec, 64] decoded weights in lane-weight order (2p+e)."""
    n=p.nrec;f=np.zeros((n,64))
    if p.rmode==RM['UNI4']:
        for P in range(32):
            x=p.wb[:,P>>2].astype(np.int64);s=(P&3)*4
            f[:,2*P]=((x>>s)&15)-8;f[:,2*P+1]=((x>>(s+16))&15)-8
        return f/16.
    eb=_ext(p.wb,p.bbits)
    for j in range(64):f[:,j]=A_*_bytes(_win(eb,step_off(j,p.bka,p.bm)))+(1024*A_+B_)
    if p.rmode in(RM['P'],RM['P2']):
        er=_ext(p.wr,p.rbits);Mb=np.repeat(p.Mb,32).astype(np.int64);Nn=np.repeat(p.Nn,32).astype(np.int64)
        f16=lambda v:np.asarray(v,np.float32).astype(np.float16).astype(np.float64)
        Ah=f16(np.float32(1.732421875)*np.float32(1.0)/Mb.astype(np.float32))
        Ch=f16((Nn.astype(np.float32)*(np.float32(1)/Mb.astype(np.float32)))*np.float32(-3.453125)+np.float32(-3.453125)-1024*Ah)
        for j in range(64):
            sb=_bytes(_win(eb,step_off(j,p.bka,p.bm)))
            if not p.v2:sr=_bytes(_win(er,step_off(j,p.rka,p.rm)))
            else:
                v=_win(er,step_off(j>>1,p.rka,p.rm));sr=_bytes(v) if j%2==0 else 510+_bytes(v,(1,-1,1,-1))
            F=(Mb*sb+Nn*sr+128)>>8
            if p.raw:
                fA=(np.float32(1.732421875)*(np.float32(1)/Mb.astype(np.float32))).astype(np.float64)
                f[:,j]=fA*(1024+F)+((Nn*fA/256*0+Nn.astype(np.float64)/Mb)*-3.453125-3.453125-1024*fA)
            else:f[:,j]=Ah*(1024+F)+Ch
        return f if p.raw else f16(f)
    if p.hasr:
        er=_ext(p.wr,p.rbits);d=np.repeat(p.dl,32)[:,None];r=np.zeros_like(f)
        if not p.v2:
            for j in range(64):r[:,j]=A_*_bytes(_win(er,step_off(j,p.rka,p.rm)))
        else:
            for P in range(32):
                v=_win(er,step_off(P,p.rka,p.rm))
                r[:,2*P]=A_*_bytes(v);r[:,2*P+1]=A_*(510+_bytes(v,(1,-1,1,-1)))
        f+=d*(r+(1024*A_+B_))
    return f
def ref_W(p):
    S,C=p.N//16,p.K//128;f=ref_vals(p).reshape(S,C,32,64);W=np.zeros((p.N,p.K))
    g=np.arange(32)>>2;t4=np.arange(32)&3
    for j in range(64):
        pp,e=j>>1,j&1;t,r=pp>>2,pp&3
        rows=np.arange(S)[:,None,None]*16+g[None,None,:]+(r&1)*8
        ks=np.arange(C)[None,:,None]*128+t*16+t4[None,None,:]*2+(r>>1)*8+e
        W[np.broadcast_to(rows,(S,C,32)),np.broadcast_to(ks,(S,C,32))]=f[...,j]
    return W
def check(p,B=3,cfg=None):
    x=torch.randn(B,p.K,device='cuda').half();y=torch.zeros(B,p.N,device='cuda');W=torch.zeros(p.N,p.K,device='cuda')
    p(x,y,cfg,wdbg=W);y2=torch.zeros(B,p.N,device='cuda');p(x,y2,cfg)
    Wr=torch.tensor(ref_W(p),device='cuda',dtype=torch.float32);ref=x.float()@Wr.T
    return ((y2-ref).norm()/ref.norm()).item(),(W-Wr).abs().max().item(),((W-Wr).norm()/Wr.norm()).item(),Wr.std().item()
