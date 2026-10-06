"""NestQuant served weights -> effective dense expert weights in the ORIGINAL basis (FP8 eval, C2).
Source = the serving repack (repack.py OUT): res/rank{r}/L{L}.pt (level-2 planes, scales, low-rank V/U2) + rank{r}.bin
(P4 record: residual planes, block words, U4). These are exactly the bytes the serve kernel reads, so the eval scores
the served weights; a new artifact version only needs a repack.
Per (layer, TP rank, expert, level) the kernel computes (moe.Expert.ref)
    g = wht(wht(x*su_g) Wg^T) * sv_g + (x Vg^T)(U2g [+U4g]),  u likewise (su_u, sv_u),  y = wht(wht(h*su_d) Wd^T)*sv_o + ...
with wht = 128-blocked orthonormal Hadamard (the down input wht at the layer's in_had_down, threads/29: 128 or 512),
so the effective weights are
    Wgate = D(sv_g) B_I Wg B_H D(su_g) + U2g^T Vg (+U4g^T Vg),  Wdown = D(sv_o) B_H Wd B_I D(su_d) + U2d^T Vd (+U4d^T Vd)
(the kernel's fp16 rounding of the rotated activations is not modelled: this eval measures the weights, bf16 math).
The rank's slice is its own intermediate block (I/tp rows of gate/up, columns of down); summing the 4 ranks' partial
outputs gives the expert output exactly, whatever intermediate permutation the encoder used.
  RankLayerEff(repack, L, rank, dev).weights(level) -> {E: (Wgate [I_r,H], Wup [I_r,H], Wdown [H,I_r])} fp32
  selftest: python nqeff.py REPACK L [rank]  -> unpack round trip + effective weights vs the kernel (MoELayer)"""
import os,sys,json,hashlib,types,torch
HERE=os.path.dirname(os.path.abspath(__file__));R=os.path.dirname(os.path.dirname(HERE))
for p in (R+'/sm120',R+'/streaming'):
    if p not in sys.path:sys.path.insert(0,p)
import moe as MO
import resident as RS

def unpack_words(p4,bits,nrec):
    """inverse of moe.pack_words: int32 packed sub-arrays -> [nrec, NW] int64 (uint32 values)."""
    n4,n2,n1,nh,T=MO.split_bits(bits);b=p4.contiguous().view(torch.uint8);dev=b.device;o=0;nw=(bits+31)//32
    w=torch.zeros(nrec,nw,dtype=torch.int64,device=dev)
    def take(nb_rec,nwords):                   # nwords little-endian words of nb_rec bytes each, per record
        nonlocal o;n=nrec*nwords*nb_rec;x=b[o:o+n].long().view(nrec,nwords,nb_rec);o+=n
        return (x<<(8*torch.arange(nb_rec,device=dev))).sum(-1)
    k=0
    if n4:w[:,:4*n4]=take(4,4*n4);k=4*n4
    if n2:w[:,k:k+2]=take(4,2);k+=2
    if n1:w[:,k]=take(4,1)[:,0];k+=1
    if nh:w[:,k]=take(2,1)[:,0]
    if T:
        A=bits-T;nbit=nrec*T;nby=(nbit+7)//8;x=b[o:o+nby].long()
        bb=((x[:,None]>>torch.arange(8,device=dev))&1).reshape(-1)[:nbit].view(nrec,T)
        w[:,A//32]|=(bb<<torch.arange(T,device=dev)).sum(-1)<<(A%32)
    return w

def _hblk(A,dim,n=128):
    """multiply A by the n-blocked orthonormal (Sylvester) Hadamard along dim (0 = left, 1 = right)."""
    Hm=MO.H128(A.device,n)
    if dim==1:return (A.reshape(A.shape[0],-1,n)@Hm).reshape(A.shape)
    return (Hm@A.reshape(-1,n,A.shape[1])).reshape(A.shape)

def sha_file(path,chunk=16<<20):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for c in iter(lambda:f.read(chunk),b''):h.update(c)
    return h.hexdigest()

class RankLayerEff:
    def __init__(s,repack,L,rank,dev='cuda',hash_=True):
        s.rf=json.load(open(f'{repack}/rank{rank}.json'));assert s.rf['format']=='nq-p4rec-v1' and str(L) in s.rf['layers'],(repack,L,rank)
        rp=f'{repack}/res/rank{rank}/L{L}.pt';s.ex,s.H,s.I=RS.load(rp,dev);s.L,s.rank,s.dev=L,rank,dev
        rb=s.rf['rec_bytes'];s.seg=s.rf['seg'];NE=s.rf['NE']
        with open(f'{repack}/rank{rank}.bin','rb') as f:
            f.seek((L-s.rf['L0'])*NE*rb);raw=f.read(NE*rb)
        assert len(raw)==NE*rb,('short record read',L,rank,len(raw))
        s.hash=dict(res=sha_file(rp) if hash_ else None,rec=hashlib.sha256(raw).hexdigest() if hash_ else None)
        s.rec=torch.frombuffer(bytearray(raw),dtype=torch.uint8).view(NE,rb).to(dev);del raw
        lj=s.rf['layers'][str(L)];s.rg={int(k):v for k,v in lj['rg'].items()};s.rd={int(k):v for k,v in lj['rd'].items()}
    def _proj(s,Es,which,level):
        """dense rotated-basis decode of projection `which` for experts Es (same residual code), batched along N
        (planes are strip-major, so experts concatenate) -> [len(Es), N, K]"""
        P0=getattr(s.ex[Es[0]],which);rk=P0.rk;N,K=(2*s.I,s.H) if which=='gu' else (s.H,s.I);n=len(Es)
        Ps=[getattr(s.ex[E],which) for E in Es];assert all(P.rk==rk for P in Ps)
        bk=int(getattr(P0,'bk',0) or 0);assert all(int(getattr(P,'bk',0) or 0)==bk for P in Ps)
        p=types.SimpleNamespace(N=N*n,K=K,rk=rk,bk=bk,z=MO.proj_sizes(N*n,K,None,rk,bk),fl=None,flags=None,
                                base=torch.cat([P.base for P in Ps]),var=torch.cat([P.var for P in Ps]))
        if bk:    # nq-res-v2 pattern-rate base: sub-array packing is per expert, so unpack each expert then concatenate
            z1=MO.proj_sizes(N,K,None,rk,bk);p.bw=torch.cat([MO.unpack_words(P.base,z1['S']*z1['C']*32,MO.rbits(bk)) for P in Ps])
        if level==4:
            z=MO.proj_sizes(N,K,None,rk);nrec=z['S']*z['C']*32;o,m=s.seg[which+'.p4']
            p.p4w=torch.cat([unpack_words(s.rec[E,o:o+m].view(torch.int32),MO.rbits(rk),nrec) for E in Es])
            o,m=s.seg[which+'.d4'];d4=torch.cat([s.rec[E,o:o+m].view(torch.int32).long() for E in Es]);p.Mb=d4&255;p.Nn=(d4>>8)&255
        return MO.dense_W(p,level).view(n,N,K)
    def experts(s,Es,level,bs=8):
        """batched expert(): {E: (Wgate, Wup, Wdown)}"""
        out={};grp={}
        for E in Es:x_=s.ex[E];grp.setdefault((x_.gu.rk,x_.dn.rk,int(getattr(x_.gu,'bk',0) or 0),int(getattr(x_.dn,'bk',0) or 0)),[]).append(E)
        for g in grp.values():
            for i in range(0,len(g),bs):
                b=g[i:i+bs];WG=s._proj(b,'gu',level);WD=s._proj(b,'dn',level)
                for j,E in enumerate(b):out[E]=s._eff(E,level,WG[j],WD[j])
        return out
    def expert(s,E,level):
        """-> (Wgate [I,H], Wup [I,H], Wdown [H,I]) fp32, original basis, this rank's intermediate block."""
        H,I=s.H,s.I;x=s.ex[E];sg=x.sc[level].float();su,svg,svu,sud,svo,suu=sg[:H],sg[H:H+I],sg[H+I:H+2*I],sg[H+2*I:H+3*I],sg[H+3*I:2*H+3*I],sg[2*H+3*I:]
        return s._eff(E,level,s._proj([E],'gu',level)[0],s._proj([E],'dn',level)[0])
    def _eff(s,E,level,Wg,Wd):
        H,I=s.H,s.I;x=s.ex[E];sg=x.sc[level].float();su,svg,svu,sud,svo,suu=sg[:H],sg[H:H+I],sg[H+I:H+2*I],sg[H+2*I:H+3*I],sg[H+3*I:2*H+3*I],sg[2*H+3*I:]
        G=_hblk(_hblk(Wg[:I],1),0)*svg[:,None]*su[None,:]
        U=_hblk(_hblk(Wg[I:],1),0)*svu[:,None]*suu[None,:]
        D=_hblk(_hblk(Wd,1,MO.had_dn(x)),0)*svo[:,None]*sud[None,:]   # down input side at in_had_down (threads/29)
        rg,rd=x.rg,x.rd
        if rg+rd:
            lr=x.lr.float();o=0
            def cut(n,shape):
                nonlocal o;t=lr[o:o+n].view(shape);o+=n;return t
            Vg=cut(rg*H,(rg,H));U2g=cut(rg*I,(rg,I));U2u=cut(rg*I,(rg,I));Vd=cut(rd*I,(rd,I));U2d=cut(rd*H,(rd,H))
            if level==4:
                o4,_=s.seg['lr4'];l4=s.rec[E,o4:o4+2*(2*rg*I+rd*H)].view(torch.float16).float()
                U2g=U2g+l4[:rg*I].view(rg,I);U2u=U2u+l4[rg*I:2*rg*I].view(rg,I);U2d=U2d+l4[2*rg*I:].view(rd,H)
            G=G+U2g.T@Vg;U=U+U2u.T@Vg;D=D+U2d.T@Vd
        return G,U,D

def _selftest(repack,L,rank=0):
    import random
    torch.manual_seed(0)
    for rk,(KA,M) in MO.RKP.items():                       # unpack round trip on random words
        bits=MO.rbits(rk);nw=(bits+31)//32;w=torch.randint(0,2**32,(4096,nw),dtype=torch.int64)
        if bits%32:w[:,-1]&=(1<<(bits%32))-1
        assert torch.equal(unpack_words(MO.pack_words(w,bits),bits,4096),w),('unpack mismatch',rk)
    print('unpack round trip OK (all residual codes)')
    import stream_engine as SE,p4rec as PR
    R_=RankLayerEff(repack,L,rank);H,I=R_.H,R_.I;dev=R_.dev;NE=256
    M=MO.MoELayer(NE,H,I,Bmax=8,dev=dev);rng=random.Random(1);E4=set(rng.sample(range(NE),128))
    slots=torch.empty(len(E4),R_.rf['rec_bytes'],dtype=torch.uint8,device=dev);lay=dict(seg=R_.seg,rec_bytes=R_.rf['rec_bytes'])
    for i,E in enumerate(sorted(E4)):slots[i].copy_(R_.rec[E]);M.table[E].copy_(PR.row(R_.ex[E],lay,slots[i].data_ptr(),MO.entry))
    for E in range(NE):
        if E not in E4:M.table[E].copy_(MO.entry(R_.ex[E],2))
    worst=0
    for t in range(6):
        B=8;sel=torch.stack([torch.tensor(rng.sample(range(NE),8)) for _ in range(B)]).to(dev)
        rw=torch.softmax(torch.randn(B,8,device=dev),1).half();x=(torch.randn(B,H,device=dev)*0.05).half()
        y=M(x,sel,rw,cfg_gu=[1,8,12],cfg_dn=[1,8,4]).float().clone();r=torch.zeros_like(y)
        for b in range(B):
            for k in range(8):
                E=int(sel[b,k]);G,U,D=R_.expert(E,4 if E in E4 else 2);xf=x[b].float()
                r[b]+=float(rw[b,k])*(D@(torch.nn.functional.silu(G@xf)*(U@xf)))
        er=((y-r).norm()/r.norm()).item();worst=max(worst,er);print(f'trial {t}: kernel vs effective dense rel {er:.2e}',flush=True)
    print(json.dumps(dict(L=L,rank=rank,worst=worst)));print('NQEFF SELFTEST','PASS' if worst<5e-3 else 'FAIL')

if __name__=='__main__':
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get('NQ_VRAM_GB','6'))/96)
    _selftest(sys.argv[1],int(sys.argv[2]),int(sys.argv[3]) if len(sys.argv)>3 else 0)
