"""nqmoe.cu vs thread 12's nq_decode on a real encoded expert (T12 artifact from nq_encode.encode_expert).
 (1) repack the T12 planes (ring streams per unit, block words, base-variant signs, per-level suh/svh) into the kernel
     layout (moe.Proj fields), gate|up fused (gate strips then up strips);
 (2) NQ_WDUMP kernel decode == nq_decode.ring_levels -> to_matrix (rotated basis), bitwise fp16, levels 2 and 4, all 3 proj;
 (3) forward: kernel MoE output (one expert, level-specific scales) vs x through nq_decode.decode_expert dense weights.
usage: verify_t12.py ARTIFACT.pt"""
import sys,types,torch;torch.cuda.set_per_process_memory_fraction(12/80)
sys.path.insert(0,'/home/coder/git/nestquant/threads/12-reference-encoder')
import nq_decode as D
import moe;from moe import *
from build import get
dev='cuda'
art=torch.load(sys.argv[1],weights_only=False,map_location='cpu')

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
g,Qg=repack(art['gate']);u,Qu=repack(art['up']);d,Qd=repack(art['down'])
gu=cat_proj(g,u);H,I=d.N,d.K;assert gu.N==2*I and gu.K==H
print(f'E: H {H} I {I}  residual K gate|up {[k for k,v in RK_OF.items() if v==gu.rk]} down {[k for k,v in RK_OF.items() if v==d.rk]}'
      f'  base_var {art["gate"]["meta"].get("base_var")}',flush=True)
def scales(L):
    pl=D.SCALE_PLANE[L];s=lambda n,k:art[n][pl][k].half().to(dev)
    return torch.cat([s('gate','suh'),s('gate','svh'),s('up','svh'),s('down','suh'),s('down','svh'),s('up','suh')])
ex=types.SimpleNamespace(gu=gu,dn=d,H=H,I=I,signs=scales(2))
assert art.get('meta',{}).get('inter_perm') is None
# (0) moe.py torch ref (independent of the kernel) == nq_decode
for L in (2,4):
    r=dense_W(gu,L,4,torch.float16);rd=dense_W(d,L,4,torch.float16)
    print(f'(0) level {L}: moe.dense_W == nq_decode  gate|up {torch.equal(r.view(torch.int16),torch.cat([Qg[L],Qu[L]]).view(torch.int16))}'
          f'  down {torch.equal(rd.view(torch.int16),Qd[L].view(torch.int16))}',flush=True)
# (2) kernel decode
Md=get(['NQ_WDUMP'])
Lr=MoELayer(1,H,I,G=4,mod=Md);sel=torch.zeros(1,8,dtype=torch.int64,device=dev)
rw=torch.zeros(1,8,device=dev).half();rw[0,0]=1;x=(torch.randn(1,H,device=dev)*0.05).half()
ok=True
for L in (2,4):
    ex.signs=scales(L);Lr.set(0,ex,L)
    wg=torch.full((2*I,H),float('nan'),device=dev).half();wd=torch.full((H,I),float('nan'),device=dev).half()
    Md.set_wdump(wg.data_ptr(),wd.data_ptr(),0);Lr(x,sel,rw);torch.cuda.synchronize();Md.set_wdump(0,0,-1)
    for nm,w,r in (('gate',wg[:I],Qg[L]),('up',wg[I:],Qu[L]),('down',wd,Qd[L])):
        eq=torch.equal(w.view(torch.int16),r.view(torch.int16));ok&=eq
        print(f'(2) level {L} {nm}: kernel == nq_decode bitwise {eq}'+('' if eq else f'  n_diff {int((w.view(torch.int16)!=r.view(torch.int16)).sum())}'),flush=True)
# (3) forward vs nq_decode dense weights (normal build)
Mn=get([]);Lf=MoELayer(1,H,I,G=4,mod=Mn)
xs=(torch.randn(4,H,device=dev)*0.05).half()
for L in (2,4):
    ex.signs=scales(L);Lf.set(0,ex,L)
    Wg,Wu,Wd=D.decode_expert(art,L,dev)
    xf=xs.float();y_ref=(torch.nn.functional.silu(xf@Wg.T)*(xf@Wu.T))@Wd.T
    y=torch.stack([Lf(xs[b:b+1],sel,rw)[0].clone() for b in range(4)])
    rel=((y-y_ref).norm()/y_ref.norm()).item();print(f'(3) level {L}: kernel forward vs nq_decode dense, rel err {rel:.2e}',flush=True)
    ok&=rel<3e-3
print('PASS' if ok else 'FAIL')
