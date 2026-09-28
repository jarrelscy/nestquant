"""nqmoe.cu vs thread 12's nq_decode on a real encoded expert (T12 artifact from nq_encode.encode_expert).
 (1) repack the T12 planes (ring streams per unit, block words, base-variant signs, per-level suh/svh) into the kernel
     layout (moe.Proj fields), gate|up fused (gate strips then up strips);
 (2) NQ_WDUMP kernel decode == nq_decode.ring_levels -> to_matrix (rotated basis), bitwise fp16, levels 2 and 4, all 3 proj;
 (3) forward: kernel MoE output (one expert, level-specific scales) vs x through nq_decode.decode_expert dense weights.
     Artifacts with the low-rank plane (T12 f128e41 lr) get it packed into table [14]/[15] (moe.Expert.set_lr); the
     dense ref (decode_expert -> apply_ocol) includes it.  Importable: load_expert(path) / scales(art, L).
usage: verify_t12.py ARTIFACT.pt"""
import sys,types,torch;torch.cuda.set_per_process_memory_fraction(12/80)
sys.path.insert(0,'/home/coder/git/nestquant/threads/12-reference-encoder')
import nq_decode as D
import moe;from moe import *
from build import get
dev='cuda'

def repack(P):
    """T12 projection planes -> Proj-like (kernel orientation N=out rows, K=in cols), + T12 rotated Q2/Q4 [N,K] fp16."""
    m=P['meta'];tk,tn=m['tk'],m['tn'];U=tk*tn
    rule=m['res_rule'];assert rule['kind']=='uniform',rule        # kernel: one residual K per projection
    K=float(rule['K']);rk=RK_OF[K];KA,MASK=RKP[rk];assert D.PATTERNS[K]==(KA,MASK)
    rl=D.ring_levels(P,dev);order=rl['order']                     # storage position i -> flat unit id u = a*tn + c
    Q={L:D.to_matrix(rl[f'Q{L}'],order,tk,tn,m['k'],m['n'],dev).T.contiguous().half() for L in (2,4)}
    rec=torch.empty(U,dtype=torch.long,device=dev)                # storage position -> kernel unit c*tk + a
    a,c=order//tn,order%tn;rec[:]=c*tk+a;inv=torch.argsort(rec)   # kernel unit -> storage position
    def lane_words(raw,bits):
        nb=8*4*bits//8;st=raw.to(dev).view(U,nb)[inv]              # [U kernel order, 8 rings * ring bytes]
        b=((st.long()[...,None]>>torch.arange(8,device=dev))&1).view(U,8,4,bits).reshape(U*32,bits)
        nw=(bits+31)//32;b=torch.nn.functional.pad(b,(0,nw*32-bits)).view(U*32,nw,32)
        return (b<<torch.arange(32,device=dev)).sum(-1)            # [U*32, nw] int64 (uint32 values)
    wb=lane_words(D._cat(P['base']['shards'],'cpu'),128)
    wr=lane_words(D._cat(P['p4']['shards'],'cpu'),rbits(rk))
    p=types.SimpleNamespace(N=m['n'],K=m['k'],rk=rk,z=proj_sizes(m['n'],m['k'],None,rk),flags=None,fl=None)
    p.base=torch.from_numpy(wb.reshape(-1).cpu().numpy().astype('uint32').view('int32')).to(dev)
    p.p4w=wr;p.p4=pack_words(wr.cpu(),rbits(rk)).to(dev)
    bw=D._cat(P['p4']['word'],'cpu').long().to(dev)[inv];p.Mb=bw&255;p.Nn=(bw>>8)&255;p.d4=(p.Mb|(p.Nn<<8)).to(torch.int32)
    p.var=None
    if m.get('base_var'):
        assert m['base_var']=='sign'
        v=D._cat(P['base']['var'],'cpu').long().to(dev).view(U,8)[inv]
        p.var=((v&1)<<torch.arange(8,device=dev)).sum(1).to(torch.uint8)
    return p,Q
def cat_proj(a,b):
    """gate|up fused along N (gate strips then up strips; same K code)"""
    assert a.rk==b.rk and a.K==b.K
    p=types.SimpleNamespace(N=a.N+b.N,K=a.K,rk=a.rk,z=proj_sizes(a.N+b.N,a.K,None,a.rk),flags=None,fl=None)
    p.base=torch.cat([a.base,b.base]);p.p4w=torch.cat([a.p4w,b.p4w]);p.p4=pack_words(p.p4w.cpu(),rbits(p.rk)).to(dev)
    p.Mb=torch.cat([a.Mb,b.Mb]);p.Nn=torch.cat([a.Nn,b.Nn]);p.d4=torch.cat([a.d4,b.d4])
    p.var=None if a.var is None else torch.cat([a.var,b.var])
    return p
def scales(art,L):
    pl=D.SCALE_PLANE[L];s=lambda n,k:art[n][pl][k].half().to(dev)
    return torch.cat([s('gate','suh'),s('gate','svh'),s('up','svh'),s('down','suh'),s('down','svh'),s('up','suh')])
def load_expert(path,art=None,verbose=True):
    """T12 artifact -> kernel expert namespace (gu, dn, signs = level-2 scales, lr plane) + rotated Q refs."""
    art=art if art is not None else torch.load(path,weights_only=False,map_location='cpu')
    assert art.get('meta',{}).get('inter_perm') is None
    g,Qg=repack(art['gate']);u,Qu=repack(art['up']);d,Qd=repack(art['down'])
    gu=cat_proj(g,u);H,I=d.N,d.K;assert gu.N==2*I and gu.K==H
    ex=types.SimpleNamespace(gu=gu,dn=d,H=H,I=I,signs=scales(art,2),lr=None,rg=0,rd=0,art=art,Qg=Qg,Qu=Qu,Qd=Qd)
    lg,lu,ld=(art[p]['base'].get('lr') for p in ('gate','up','down'))
    assert (lg is None)==(lu is None),'kernel: gate/up lr both present or both absent'
    if lg is not None or ld is not None:
        if lg is not None:   # kernel stores one V for gate|up (production: shared input Gram -> shared_V)
            Vg,U2g,U2u,U4g,U4u=lg['V'],lg['U2'],lu['U2'],art['gate']['p4']['lr']['U4'],art['up']['p4']['lr']['U4']
            if not art['up']['meta']['lr'].get('shared_V'):   # separate V (older encodes): stack [V_g; V_u], zero-pad U
                rg_,ru_=Vg.shape[0],lu['V'].shape[0];assert rg_+ru_<=4,'kernel: r_gate + r_up <= 4 when V is not shared'
                z=lambda r,n:torch.zeros(r,n,dtype=U2g.dtype)
                Vg=torch.cat([Vg,lu['V']]);U2g=torch.cat([U2g,z(ru_,I)]);U4g=torch.cat([U4g,z(ru_,I)])
                U2u=torch.cat([z(rg_,I),U2u]);U4u=torch.cat([z(rg_,I),U4u])
            else:assert torch.equal(lu['V'],lg['V'])
        else:Vg=torch.zeros(0,H);U2g=U2u=U4g=U4u=torch.zeros(0,I)
        if ld is not None:Vd,U2d,U4d=ld['V'],ld['U2'],art['down']['p4']['lr']['U4']
        else:Vd=torch.zeros(0,I);U2d=U4d=torch.zeros(0,H)
        moe.Expert.set_lr(ex,Vg,U2g,U2u,U4g,U4u,Vd,U2d,U4d)
    if verbose:print(f'E: H {H} I {I}  residual K gate|up {[k for k,v in RK_OF.items() if v==gu.rk]} down {[k for k,v in RK_OF.items() if v==d.rk]}'
          f'  base_var {art["gate"]["meta"].get("base_var")}  lr rank gate/up {ex.rg} down {ex.rd}',flush=True)
    return ex
if __name__=='__main__':
    ex=load_expert(sys.argv[1]);art=ex.art;H,I,gu,d=ex.H,ex.I,ex.gu,ex.dn
    # (0) moe.py torch ref (independent of the kernel) == nq_decode
    for L in (2,4):
        r=dense_W(gu,L,4,torch.float16);rd=dense_W(d,L,4,torch.float16)
        print(f'(0) level {L}: moe.dense_W == nq_decode  gate|up {torch.equal(r.view(torch.int16),torch.cat([ex.Qg[L],ex.Qu[L]]).view(torch.int16))}'
              f'  down {torch.equal(rd.view(torch.int16),ex.Qd[L].view(torch.int16))}',flush=True)
    # (2) kernel decode
    Md=get(['NQ_WDUMP'])
    Lr=MoELayer(1,H,I,G=4,mod=Md);sel=torch.zeros(1,8,dtype=torch.int64,device=dev)
    rw=torch.zeros(1,8,device=dev).half();rw[0,0]=1;x=(torch.randn(1,H,device=dev)*0.05).half()
    ok=True
    for L in (2,4):
        ex.signs=scales(art,L);Lr.set(0,ex,L)
        wg=torch.full((2*I,H),float('nan'),device=dev).half();wd=torch.full((H,I),float('nan'),device=dev).half()
        Md.set_wdump(wg.data_ptr(),wd.data_ptr(),0);Lr(x,sel,rw);torch.cuda.synchronize();Md.set_wdump(0,0,-1)
        for nm,w,r in (('gate',wg[:I],ex.Qg[L]),('up',wg[I:],ex.Qu[L]),('down',wd,ex.Qd[L])):
            eq=torch.equal(w.view(torch.int16),r.view(torch.int16));ok&=eq
            print(f'(2) level {L} {nm}: kernel == nq_decode bitwise {eq}'+('' if eq else f'  n_diff {int((w.view(torch.int16)!=r.view(torch.int16)).sum())}'),flush=True)
    # (3) forward vs nq_decode dense weights (normal build)
    Mn=get([]);Lf=MoELayer(1,H,I,G=4,mod=Mn)
    xs=(torch.randn(4,H,device=dev)*0.05).half()
    for L in (2,4):
        ex.signs=scales(art,L);Lf.set(0,ex,L)
        Wg,Wu,Wd=D.decode_expert(art,L,dev)
        xf=xs.float();y_ref=(torch.nn.functional.silu(xf@Wg.T)*(xf@Wu.T))@Wd.T
        y=torch.stack([Lf(xs[b:b+1],sel,rw)[0].clone() for b in range(4)])
        rel=((y-y_ref).norm()/y_ref.norm()).item();print(f'(3) level {L}: kernel forward vs nq_decode dense, rel err {rel:.2e}',flush=True)
        ok&=rel<3e-3
    print('PASS' if ok else 'FAIL')
