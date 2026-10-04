"""NestQuant grouped MoE layer: pools, device table, launch wrapper, dense reference decode (thread 15 RM_P spec).
Plane layout: see nqmoe.cu header."""
import os,torch,numpy as np
from build import get
M=get()
TBL_W=20
A_=float(np.array([0x1eee],np.uint16).view(np.float16)[0]);B_=float(np.array([0xc931],np.uint16).view(np.float16)[0])

# residual window patterns (table code -> (KA, MASK)); LSB-first period-16 fractional steps (thread 15)
RKP={0:(2,0),1:(1,0xEEEE),2:(2,0xAAAA),3:(2,0x8888),4:(3,0),5:(1,0xAAAA),6:(1,0xFFFE),7:(2,0x9248),8:(1,0xFEFE),
     9:(2,0xD5AA)}                                       # 9: K=2.5625 (2, bres(9)), threads/35 b1.75 down residual
RK_OF={2:0,1.75:1,2.5:2,2.25:3,3:4,1.5:5,1.9375:6,2.3125:7,1.875:8,2.5625:9}
# Base K code (nq-res-v2, table [19]): same numbering as RKP. 0 = today's K=2 ring base (uint4 per record, == the
# sub-array layout at 128 bits); 1 = K=1.75 (1,0xEEEE) pattern-rate base (threads/35), 112 bits per record in the P4
# sub-array layout (uint2 | uint | ushort). Spec: sm120/NQ_RES_V2.md.
popc=lambda m:bin(m).count('1')
def rbits(rk):KA,M=RKP[rk];return 4*(16*KA+popc(M))
def step_off(p,KA,MASK):
    per=[sum(KA+((MASK>>i)&1) for i in range(r)) for r in range(16)]
    p=torch.as_tensor(p);return (p>>4)*(16*KA+popc(MASK))+torch.tensor(per,device=p.device)[p&15]
def split_bits(bits):n4=bits//128;r=bits%128;return n4,r//64,(r%64)//32,(r%32)//16,bits%16   # + tail bits T

def proj_sizes(N,K,nm=None,rk=0,bk=0):
    """bytes per plane (flags: int64 per strip). nm=None: dense P4; else nm flagged chunks per strip."""
    S,C=N//16,K//128;R=C if nm is None else nm
    return dict(S=S,C=C,nm=nm,rk=rk,bk=bk,base=S*C*32*rbits(bk)//8+(4 if rbits(bk)%16 else 0),p4=S*R*32*rbits(rk)//8+(4 if rbits(rk)%16 else 0),d4=S*R*4,flags=0 if nm is None else S*8,nrec_r=S*R*32)

def make_flags(S,C,nm,gen,grp=8):
    """nm flagged chunks per strip; identical across each group of `grp` strips (128x128 block granularity)."""
    ng=(S+grp-1)//grp
    idx=torch.rand(ng,C,generator=gen).argsort(1)[:,:nm]
    fl=torch.zeros(ng,C,dtype=torch.bool);fl.scatter_(1,idx,True)
    fl=fl.repeat_interleave(grp,0)[:S]
    return (fl.long()<<torch.arange(C)).sum(1),fl        # C <= 63

def pack_words(w,bits):
    """w [nrec, NW] int64 (uint32 values) -> int32 tensor of the sub-array layout (uint4 x n4 | uint2 | uint | ushort)."""
    n4,n2,n1,nh,T=split_bits(bits);parts=[];k=0
    def as_bytes(t,nb):                                   # int64 values -> little-endian bytes, nb bytes each
        return torch.stack([(t>>(8*i))&255 for i in range(nb)],-1).to(torch.uint8).reshape(-1)
    if n4:parts.append(as_bytes(w[:,:4*n4],4));k=4*n4
    if n2:parts.append(as_bytes(w[:,k:k+2],4));k+=2
    if n1:parts.append(as_bytes(w[:,k],4));k+=1
    if nh:parts.append(as_bytes(w[:,k]&0xFFFF,2))
    if T:                                                 # tail: T bits per record, bit-packed over records (+4 B pad)
        A=bits-T;t=(w[:,A//32]>>(A%32))&((1<<T)-1)
        b=((t[:,None]>>torch.arange(T))&1).reshape(-1);b=torch.cat([b,torch.zeros(32,dtype=b.dtype)])
        b=b[:b.numel()//8*8].view(-1,8);parts.append((b<<torch.arange(8)).sum(1).to(torch.uint8))
    return torch.cat(parts).view(torch.int32)

def unpack_words(t,nrec,bits):
    """inverse of pack_words: int32 sub-array layout of nrec records -> [nrec, NW] int64 (uint32 values)."""
    n4,n2,n1,nh,T=split_bits(bits);nw=(bits+31)//32;b=t.contiguous().view(torch.uint8).long();o=0
    w=torch.zeros(nrec,nw,dtype=torch.int64,device=t.device)
    def take(nwords,nb):
        nonlocal o;x=b[o:o+nrec*nwords*nb].view(nrec,nwords,nb);o+=nrec*nwords*nb
        return (x<<(8*torch.arange(nb,device=t.device))).sum(-1)
    k=0
    if n4:w[:,:4*n4]=take(4*n4,4);k=4*n4
    if n2:w[:,k:k+2]=take(2,4);k+=2
    if n1:w[:,k]=take(1,4)[:,0];k+=1
    if nh:w[:,k]=take(1,2)[:,0]
    if T:
        A=bits-T;bb=b[o:o+(nrec*T+7)//8];bits_=((bb[:,None]>>torch.arange(8,device=t.device))&1).reshape(-1)[:nrec*T].view(nrec,T)
        w[:,A//32]|=(bits_<<torch.arange(T,device=t.device)).sum(-1)<<(A%32)
    return w

MB_LIST=None
def all_MbN():
    """every valid block word (Mb, N): 1<=Mb<=255, 0<=N<=255, Mb+N<=257"""
    global MB_LIST
    if MB_LIST is None:
        MB_LIST=torch.tensor([(m,n) for m in range(1,256) for n in range(0,min(255,257-m)+1)])
    return MB_LIST

class Proj:
    """Random packed planes of one projection of one expert (separate tensors so they can live in pools).
    base: int32 [S*C*32*4] (uint4 per record). p4: int32 view of the packed residual sub-arrays. d4: int32 Mb|N<<8 per block.
    mbn: 'rand' (realistic delta ~0.1-0.8 plus random extremes) or ('exh', offset): cycle through every valid (Mb, N)."""
    def __init__(s,N,K,gen,nm=None,rk=0,mbn='rand',var=False,bk=0):
        z=proj_sizes(N,K,nm,rk,bk);s.z=z;s.N,s.K,s.rk,s.bk=N,K,rk,bk
        S,C=z['S'],z['C'];R=C if nm is None else nm;nr=S*R*32;bits=rbits(rk);nw=(bits+31)//32
        ri=lambda n:torch.randint(-2**31,2**31-1,(n,),generator=gen,dtype=torch.int32)
        if bk==0:s.base=ri(S*C*32*4)                     # (bk = 0: unchanged RNG stream)
        else:
            bb=rbits(bk);wb=torch.randint(0,2**32,(S*C*32,(bb+31)//32),generator=gen,dtype=torch.int64)
            if bb%32:wb[:,-1]&=(1<<(bb%32))-1
            s.bw=wb;s.base=pack_words(wb,bb)
        w=torch.randint(0,2**32,(nr,nw),generator=gen,dtype=torch.int64)
        if bits%32:w[:,-1]&=(1<<(bits%32))-1
        s.p4w=w;s.p4=pack_words(w,bits)
        nb=S*R
        if mbn=='rand':
            Mb=torch.randint(120,256,(nb,),generator=gen);Nn=(torch.rand(nb,generator=gen)*0.8*Mb).round().long()
            ext=torch.rand(nb,generator=gen)<0.05;Mb2=torch.randint(1,256,(nb,),generator=gen)
            N2=(torch.rand(nb,generator=gen)*(torch.clamp(257-Mb2,max=255)+1)).floor().long()
            Mb=torch.where(ext,Mb2,Mb);Nn=torch.where(ext,N2,Nn);Nn=torch.minimum(Nn,257-Mb)
        else:
            L=all_MbN();idx=(torch.arange(nb)+mbn[1])%len(L);Mb,Nn=L[idx,0],L[idx,1]
        s.Mb,s.Nn=Mb,Nn;s.d4=(Mb|(Nn<<8)).to(torch.int32)
        s.var=torch.randint(0,256,(S*C,),generator=gen,dtype=torch.int64).to(torch.uint8) if var else None   # base variant signs
        if nm is None:s.flags=None;s.fl=None
        else:s.flags,s.fl=make_flags(S,C,nm,gen)
    def to(s,dev):
        for k in ['base','p4','d4','flags','p4w','Mb','Nn','var','bw']:
            if getattr(s,k,None) is not None:setattr(s,k,getattr(s,k).to(dev))
        return s

K0=np.float32(-3.453125)
RCP=torch.tensor(np.concatenate([[0],np.float32(1)/np.arange(1,256,dtype=np.float32)]).astype(np.float32))
def _bits(w,nbits):
    """[n, NW] int64 -> [n, nbits] uint8 LSB-first"""
    return ((w[...,None]>>torch.arange(32,device=w.device))&1).to(torch.uint8).reshape(w.shape[0],-1)[:,:nbits]
def _S(st):
    x=(st*0x83DCD12D)&0xFFFFFFFF
    return (x&255)+((x>>8)&255)+((x>>16)&255)+((x>>24)&255)
def lane_sums(w,nbits,KA,MASK,G,chunk=1<<14):
    """ring streams of G lanes (records lane-minor); S(state) at every lane weight -> [nrec, 64] int64 (ref15_spec.states + S)."""
    n=w.shape[0];out=torch.empty(n,64,dtype=torch.int64,device=w.device)
    L=G*nbits;off=step_off(torch.arange(64*G,device=w.device),KA,MASK)
    assert int(step_off(torch.tensor(64*G),KA,MASK))==L
    idx=(off[:,None]+torch.arange(16,device=w.device))%L;pw=(1<<torch.arange(16,device=w.device))
    for a in range(0,n,chunk):
        b=_bits(w[a:a+chunk],nbits).reshape(-1,L)                   # [rings, L]
        st=(b[:,idx].long()*pw).sum(-1)                               # [rings, 64G]
        out[a:a+chunk]=_S(st).reshape(-1,64)
    return out
def fold_vals(Sb,Sr,Mb,N,sg=1.0):
    """ref15_spec.fold in torch (fp32 ops in spec order), Mb/N broadcast [..,1]"""
    rc=RCP.to(Sb.device)[Mb]
    Ah=(np.float32(1.732421875)*rc).half()
    t=(N.float()*rc)*np.float32(K0);t=t+np.float32(K0);C=(t-1024*Ah.float()).half()
    F=(Mb*Sb+N*Sr+128)>>8
    return ((sg*Ah.double())*(1024+F)+sg*C.double()).half()     # sg = base variant sign (+-1, exact in fp16)
def lane_vals(p,level,G):
    """decoded fp16 weights [S, C, 32, 64] in lane-weight order"""
    z=p.z;S,C=z['S'],z['C'];dev=p.base.device
    bk=getattr(p,'bk',0)
    if bk==0:wb=(p.base.long()&0xFFFFFFFF).view(-1,4)
    else:wb=p.bw if getattr(p,'bw',None) is not None else unpack_words(p.base,S*C*32,rbits(bk))
    Sb=lane_sums(wb,rbits(bk),*RKP[bk],G).view(S,C,32,64)
    sg=1.0
    if getattr(p,'var',None) is not None:
        assert G==4;bit=(p.var.long().view(S,C,1)>>(torch.arange(32,device=dev)>>2))&1;sg=(1-2*bit).double()[...,None]
    q2=((sg*A_)*(1024+Sb.double())+sg*B_).half()
    if level==2:return q2
    KA,M=RKP[p.rk];R=C if p.fl is None else z['nm']
    p4w=p.p4w if getattr(p,'p4w',None) is not None else unpack_words(p.p4,S*R*32,rbits(p.rk))
    Sr=lane_sums(p4w,rbits(p.rk),KA,M,G).view(S,R,32,64)
    Mb=p.Mb.view(S,R);Nn=p.Nn.view(S,R)
    if p.fl is None:on=torch.ones(S,C,dtype=torch.bool,device=dev);rank=torch.arange(C,device=dev)[None].expand(S,C)
    else:on=p.fl.to(dev);rank=(torch.cumsum(on.long(),1)-on.long()).clamp(max=R-1)
    si=torch.arange(S,device=dev)[:,None].expand(S,C)
    q4=fold_vals(Sb,Sr[si,rank],Mb[si,rank][...,None,None],Nn[si,rank][...,None,None],sg)
    return torch.where(on[...,None,None],q4,q2)
def dense_W(p,level,G=4,dtype=torch.float32):
    """Independent dense decode of one projection at level 2 or 4 -> [N,K] (bit-exact fp16 values)."""
    f=lane_vals(p,level,G);S,C=f.shape[:2];dev=f.device
    W=torch.zeros(p.N,p.K,device=dev,dtype=torch.float16)
    lane=torch.arange(32,device=dev);g=lane>>2;t4=lane&3
    sI=torch.arange(S,device=dev)[:,None,None];cI=torch.arange(C,device=dev)[None,:,None]
    for j in range(64):
        pp,e=j>>1,j&1;t,r_=pp>>2,pp&3
        rows=(sI*16+g[None,None]+(r_&1)*8).expand(S,C,32);ks=(cI*128+t*16+t4[None,None]*2+(r_>>1)*8+e).expand(S,C,32)
        W[rows,ks]=f[...,j]
    return W.to(dtype)

def H128(dev='cuda',n=128):
    """normalized Sylvester Hadamard of order n (default 128)"""
    H=torch.ones(1,1)
    while H.shape[0]<n:H=torch.cat([torch.cat([H,H],1),torch.cat([H,-H],1)],0)
    return (H/n**0.5).to(dev)
def had_dn(ex):
    """down-projection input Hadamard width (threads/29 in_had_down; absent = 128)"""
    return int(getattr(ex,'had_dn',128) or 128)

class Expert:
    def __init__(s,H,I,seed,nm_gu=None,nm_dn=None,dev='cuda',rk_gu=0,rk_dn=0,mbn='rand',var=False,bk_gu=0,bk_dn=0):
        gen=torch.Generator().manual_seed(seed)
        s.gu=Proj(2*I,H,gen,nm_gu,rk_gu,mbn,var,bk_gu).to(dev);s.dn=Proj(H,I,gen,nm_dn,rk_dn,mbn if mbn=='rand' else ('exh',mbn[1]+(2*I//16)*(H//128)),var,bk_dn).to(dev)
        sg=torch.randint(0,2,(2*H+3*I,),generator=gen)*2-1;su_u=torch.randint(0,2,(H,),generator=gen)*2-1
        s.signs=torch.cat([sg,su_u]).half().to(dev)      # [H su_g | I sv_g | I sv_u | I su_d | H sv_o | H su_u]
        s.H,s.I=H,I;s.lr=s.lr4=None;s.rg=s.rd=0
    def set_lr(s,Vg,U2g,U2u,U4g,U4u,Vd,U2d,U4d):
        """T12 low-rank plane (fp16): Vg [rg,H], U*g/U*u [rg,I], Vd [rd,I], U*d [rd,H] -> table [14]/[15] buffers."""
        f=lambda *t:torch.cat([x.reshape(-1) for x in t]).half().contiguous()
        s.rg,s.rd=Vg.shape[0],Vd.shape[0];s.lrT=(Vg,U2g,U2u,U4g,U4u,Vd,U2d,U4d)
        if s.rg+s.rd==0:s.lr=s.lr4=None;return s
        s.lr=f(Vg,U2g,U2u,Vd,U2d).to(s.signs.device);s.lr4=f(U4g,U4u,U4d).to(s.signs.device);return s
    def bytes(s,level):
        b=0
        for p in (s.gu,s.dn):
            b+=p.base.numel()*4+(0 if getattr(p,'var',None) is None else p.var.numel())
            if level>=4:b+=(p.p4.numel()+p.d4.numel())*4+(0 if p.flags is None else p.flags.numel()*8)
        return b
    def ref(s,x,level,G=4):
        """x [T,H] fp32 -> [T,H] fp32, through the dense-decoded weights."""
        H,I=s.H,s.I;Hm=H128(x.device);wht=lambda v,M=Hm:(v.view(*v.shape[:-1],-1,M.shape[0])@M).view(v.shape)
        Hd=H128(x.device,had_dn(s))
        sg=s.signs.float();su,svg,svu,sud,svo,suu=sg[:H],sg[H:H+I],sg[H+I:H+2*I],sg[H+2*I:H+3*I],sg[H+3*I:2*H+3*I],sg[2*H+3*I:]
        Wg=dense_W(s.gu,level,G);Wd=dense_W(s.dn,level,G)
        xg=wht(x*su).half().float();xu=wht(x*suu).half().float();a=torch.cat([xg@Wg[:I].T,xu@Wg[I:].T],1)
        g=wht(a[:,:I])*svg;u=wht(a[:,I:])*svu
        if s.lr is not None:
            Vg,U2g,U2u,U4g,U4u,Vd,U2d,U4d=[t.float() for t in s.lrT];z=x@Vg.T
            g=g+z@U2g+(z@U4g if level==4 else 0);u=u+z@U2u+(z@U4u if level==4 else 0)
        sw=torch.nn.functional.silu(g)*u
        h=wht(sw*sud,Hd).half().float();y=wht(h@Wd.T)*svo
        if s.lr is not None:z=sw@Vd.T;y=y+z@U2d+(z@U4d if level==4 else 0)
        return y

def base_code(p):
    """base K code of a projection (RKP numbering; absent = 0 = K=2 ring base)"""
    return int(getattr(p,'bk',0) or 0)

def entry(ex,level):
    e=torch.zeros(TBL_W,dtype=torch.int64)
    e[0]=level
    for off,p in ((1,ex.gu),(5,ex.dn)):
        for i,k in enumerate(['base','p4','d4','flags']):
            t=getattr(p,k);e[off+i]=0 if t is None else t.data_ptr()
    e[9]=ex.signs.data_ptr();e[10]=ex.gu.rk;e[11]=ex.dn.rk
    for i,p in ((12,ex.gu),(13,ex.dn)):
        v=getattr(p,'var',None);e[i]=0 if v is None else v.data_ptr()
    lr=getattr(ex,'lr',None)
    if lr is not None:e[14]=lr.data_ptr();e[15]=ex.lr4.data_ptr();e[16]=ex.rg;e[17]=ex.rd
    w=had_dn(ex);e[18]=0 if w==128 else w
    e[19]=base_code(ex.gu)|base_code(ex.dn)<<8           # nq-res-v2 base K codes (0 = K=2 ring base, as before)
    return e

class MoELayer:
    """Device table [E,TBL_W] + workspace for up to Bmax tokens. Table entries are flipped in place (graph-safe)."""
    def __init__(s,E,H,I,nm_gu=0,nm_dn=0,Bmax=4,topk=8,G=4,dev='cuda',mod=None):
        """nm_*: flagged chunks per strip for experts in mask mode (ignored for dense experts)."""
        s.M=mod or M;s.E,s.H,s.I,s.nm_gu,s.nm_dn,s.G=E,H,I,nm_gu,nm_dn,G
        s.table=torch.zeros(E,TBL_W,dtype=torch.int64,device=dev)
        S=Bmax*topk
        s.acc_gu=torch.zeros(S,2*I,dtype=torch.float32,device=dev);s.h=torch.zeros(S,I,dtype=torch.float16,device=dev);s.acc_d=torch.zeros(S,H,dtype=torch.float32,device=dev)
        s.cnt_gu=torch.zeros(S*(I//128),dtype=torch.int32,device=dev);s.cnt_d=torch.zeros(H//128,dtype=torch.int32,device=dev);s.wq=torch.zeros(4,dtype=torch.int32,device=dev)
        s.zd=torch.zeros(S*(I//128)*4,dtype=torch.float32,device=dev);s.cnt_h=torch.zeros(S*(I//128),dtype=torch.int32,device=dev)
        s.out=torch.zeros(Bmax,H,dtype=torch.float32,device=dev)
        s.cfg_gu=[1,8,3];s.cfg_dn=[1,8,2];s.hits_ptr=0   # set to a (host-mapped) int32 [E] pointer to export routing hits
    def set(s,e,ex,level):
        w=had_dn(ex);assert w in (128,512) and s.I%w==0,f'in_had_down {w} must be 128 or 512 and divide I={s.I}'
        if getattr(ex,'lr',None) is not None:assert ex.rg<=4 and ex.rd<=4 and ex.lr.dtype==torch.float16
        bks=(base_code(ex.gu),base_code(ex.dn))
        if bks!=(0,0):                                    # kernels without bk_codes() decode only the K=2 base
            if not hasattr(s,'bkm'):s.bkm=s.M.bk_codes() if hasattr(s.M,'bk_codes') else [1,1]
            assert s.bkm[0]>>bks[0]&1 and s.bkm[1]>>bks[1]&1,f'base K code gu {bks[0]} / dn {bks[1]} not compiled (bk_codes {s.bkm})'
        if level==4:
            if not hasattr(s,'rkm'):s.rkm=s.M.rk_codes() if hasattr(s.M,'rk_codes') else [255,255]
            assert s.rkm[0]>>ex.gu.rk&1 and s.rkm[1]>>ex.dn.rk&1,f'residual K code gu {ex.gu.rk} / dn {ex.dn.rk} not compiled (rk_codes {s.rkm})'
        s.table[e].copy_(entry(ex,level).to(s.table.device),non_blocking=False)
    def __call__(s,x,sel,rw,out=None,force_level=0,which=3,cfg_gu=None,cfg_dn=None,table=None):
        """table: optional [E,TBL_W] override (default s.table), e.g. NQ_LMPF ring rows"""
        assert x.dtype==torch.float16 and rw.dtype==torch.float16 and sel.dtype==torch.int64,(x.dtype,rw.dtype,sel.dtype)
        out=s.out[:x.shape[0]] if out is None else out
        s.M.moe_forward(x,sel,rw,s.table if table is None else table,out,s.acc_gu,s.h,s.acc_d,s.cnt_gu,s.cnt_d,s.wq,s.I,s.nm_gu,s.nm_dn,
                      cfg_gu or s.cfg_gu,cfg_dn or s.cfg_dn,s.G,force_level,which,s.hits_ptr,s.zd,s.cnt_h)
        return out

_PF={}
def pf_scratch(dev,H,I,R,G):
    """Prefill scratch shared by every layer on `dev` (fixed size, allocated on first use so vLLM's profile run counts it):
    W_gu/W_dn fp16 for G experts (G*3*H*I*2 B) + per-pair rows (R rows: xg|xu fp16 = y fp32 2H*2 B, acc_g|acc_u fp32 2I*4 B,
    h fp16 I*2 B, z_gu/z_dn 32 B)."""
    k=(str(dev),H,I,R,G)
    if k not in _PF:
        f16,f32=torch.float16,torch.float32
        xy=torch.empty(2*R*H,dtype=f16,device=dev)   # xg | xu, then (after the gate|up GEMMs and pf_mid) reused as y fp32 [R][H]
        _PF[k]=dict(Wgu=torch.empty(G*2*I*H,dtype=f16,device=dev),Wdn=torch.empty(G*H*I,dtype=f16,device=dev),
                    xg=xy[:R*H].view(R,H),xu=xy[R*H:].view(R,H),y=xy.view(f32).view(R,H),acc=torch.empty(2,R,I,dtype=f32,device=dev),
                    h=torch.empty(R,I,dtype=f16,device=dev),z=torch.empty(2,R,4,dtype=f32,device=dev))
    return _PF[k]
def pf_bytes(H,I,R,G):return G*3*H*I*2+R*(4*H+8*I+2*I+32)

PF_BM=int(os.environ.get('NQ_PF_BM','64'))
PG=None
if os.environ.get('NQ_PF_GEMM','triton')=='triton':
    try:import pf_gemm as PG
    except Exception:PG=None   # no triton: per-expert cuBLAS calls
def prefill(s,x,sel,rw,out=None,R=None,G=None,table=None):
    """T > Bmax tokens: each routed expert decoded once (level from the live device table row, after the mailbox apply),
    then grouped fp16 GEMMs with fp32 outputs; same math as moe_forward (fp32 accumulation order differs). One host sync
    (expert counts). Not graph-capturable. Exports routing hits like the decode kernel (picks with rw != 0)."""
    assert x.dtype==torch.float16 and rw.dtype==torch.float16 and sel.dtype==torch.int64,(x.dtype,rw.dtype,sel.dtype)
    T,k=sel.shape;H,I,E=s.H,s.I,s.E;dev=x.device;M=s.M
    tb=s.table if table is None else table             # NQ_LMPF: override table with ring rows
    R=R or int(os.environ.get('NQ_PF_ROWS','8192'));G=G or int(os.environ.get('NQ_PF_G','16'))
    S=pf_scratch(dev,H,I,R,G)
    out=torch.zeros(T,H,dtype=torch.float32,device=dev) if out is None else out.zero_()
    flat=sel.reshape(-1);rwf=rw.reshape(-1)
    order=torch.argsort(flat,stable=True)
    pe=flat[order].int();pt=(order//k).int();prw=rwf[order].contiguous()
    one=torch.ones_like(flat,dtype=torch.int32)
    cnt=torch.zeros(E,dtype=torch.int32,device=dev).scatter_add_(0,flat,one)   # (torch.bincount would sync on its max)
    if s.hits_ptr:M.pf_hits(torch.zeros(E+1,dtype=torch.int32,device=dev).scatter_add_(0,torch.where(rwf!=0,flat,E),one)[:E],s.hits_ptr)
    cnt=cnt.cpu().tolist()
    ex=[e for e in range(E) if cnt[e]];start=[0]*E;o=0
    for e in range(E):start[e]=o;o+=cnt[e]
    exl=torch.tensor(ex,dtype=torch.int32).to(dev,non_blocking=True)
    Wgu=S['Wgu'].view(G,2*I,H);Wdn=S['Wdn'].view(G,H,I);f32=torch.float32
    mmo=torch.ops.aten.mm.dtype_out
    for g0 in range(0,len(ex),G):
        grp=ex[g0:g0+G];M.pf_decode(tb,exl[g0:g0+len(grp)],S['Wgu'],S['Wdn'],H,I,s.nm_gu,s.nm_dn)
        a=start[grp[0]];b=start[grp[-1]]+cnt[grp[-1]]
        for c in range(a,b,R):
            d=min(b,c+R);n=d-c;pts,pes,prws=pt[c:d],pe[c:d],prw[c:d]
            xg,xu,y=S['xg'][:n],S['xu'][:n],S['y'][:n];ag,au=S['acc'][0,:n],S['acc'][1,:n];h=S['h'][:n];zg,zd=S['z'][0,:n],S['z'][1,:n]
            M.pf_pre(x,pts,pes,tb,S['xg'],S['xu'],zg,I)
            segs=[]
            for j,e in enumerate(grp):
                u=max(start[e],c)-c;v=min(start[e]+cnt[e],d)-c
                if v>u:segs.append((j,u,v))
            if PG:
                tt,nt=PG.tiles(segs,PF_BM,dev)
                PG.gmm(xg,Wgu[:,:I],ag,tt,nt,I,H,BM=PF_BM);PG.gmm(xu,Wgu[:,I:],au,tt,nt,I,H,BM=PF_BM)
            else:
                for j,u,v in segs:
                    mmo(xg[u:v],Wgu[j,:I].t(),f32,out=ag[u:v]);mmo(xu[u:v],Wgu[j,I:].t(),f32,out=au[u:v])
            M.pf_mid(ag,au,zg,pes,tb,h,zd,H,I)
            if PG:PG.gmm(h,Wdn,y,tt,nt,H,I,BM=PF_BM)
            else:
                for j,u,v in segs:mmo(h[u:v],Wdn[j].t(),f32,out=y[u:v])
            M.pf_post(y,zd,pts,pes,prws,tb,out,I)
    return out
MoELayer.prefill=prefill

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

def moe_ref(experts,levels,x,sel,rw,G=4):
    y=torch.zeros(x.shape[0],x.shape[1],device=x.device);xf=x.float();cache={}
    for b in range(x.shape[0]):
        for k in range(sel.shape[1]):
            e=int(sel[b,k]);w=float(rw[b,k])
            if levels[e]<=0 or w==0:continue
            y[b]+=w*experts[e].ref(xf[b:b+1],levels[e],G)[0]
    return y
