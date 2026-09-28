"""Low-rank plane (T12 f128e41 lr) on real artifacts: nqmoe.cu vs thread 12's nq_decode.decode_expert.
 (1) per expert, levels 2 and 4: kernel forward vs (a) decode_expert dense fp32 forward (floor ~9e-4 = the kernel's fp16
     activation rounding, same as without lr), (b) moe.Expert.ref (same fp16 rounding points as the kernel; its lr term
     is z = x V^T, y += z U, the factored form of T12's W + f32(U)^T f32(V)), also (c) with every U2/U4 scaled by 64
     (exact in fp16) so the lr term dominates the output: the lr path itself is then what is being compared.
     The real lr term is only 5e-4..1e-2 of |y|, i.e. comparable to the fp16 rounding noise, so differencing two kernel
     runs (lr on / off) does not isolate it; the x64 boost does.
 (2) mixed levels in one launch: 2 experts (E168 / E169) at L2 / L4 and L4 / L2, grid and persistent configs, B = 4;
 (3) TP8: the tp{s}.pt shard lr fields reassemble to the full artifact; per rank, a shard-shape (I = 256) expert with the
     real shard lr fields vs Expert.ref; and the sum over ranks of the kernel's down-lr contribution (partial z per rank,
     replicated U): per rank z_s = projection of (kernel - kernel with down lr off) onto U_d; sum_s z_s == (concatenated
     SwiGLU output) @ V_d_full^T, i.e. y_s = partial + z_s U_d is linear in z_s and the row-parallel output all-reduce
     covers it with no extra reduce.  U_d x1024 (exact): at L4 U2+U4 nearly cancels (|U2+U4| = 0.23 |U2|), the real term
     is ~1e-6 |y|, below the run-to-run fp32 atomic-order noise of the base down GEMM (~1e-6 |y|).
usage: verify_lr.py [ROOT=/tmp/nestquant/12-reference-encoder/smoke_lr] [L=30]"""
import sys,torch;torch.cuda.set_per_process_memory_fraction(12/80)
import verify_t12 as V;from verify_t12 import D,load_expert,scales
from moe import *
dev='cuda'
root=sys.argv[1] if len(sys.argv)>1 else '/tmp/nestquant/12-reference-encoder/smoke_lr';LY=int(sys.argv[2]) if len(sys.argv)>2 else 30
EIDS=(168,169);torch.manual_seed(0);ok=True
rel=lambda a,b:((a-b).norm()/b.norm()).item()
def strip_lr(art):
    return {p:(dict(v,base={k:w for k,w in v['base'].items() if k!='lr'},p4={k:w for k,w in v['p4'].items() if k!='lr'})
               if p in ('gate','up','down') else v) for p,v in art.items()}
def boost(art,sc=64.):
    out=dict(art)
    for p in ('gate','up','down'):
        if art[p]['base'].get('lr') is None:continue
        out[p]=dict(art[p],base=dict(art[p]['base'],lr=dict(art[p]['base']['lr'],U2=art[p]['base']['lr']['U2']*sc)),
                    p4=dict(art[p]['p4'],lr=dict(U4=art[p]['p4']['lr']['U4']*sc)))
    return out
def dense_fwd(art,L,x):
    Wg,Wu,Wd=D.decode_expert(art,L,dev);return (torch.nn.functional.silu(x@Wg.T)*(x@Wu.T))@Wd.T
def run(Lk,ex,L,x,e=0,lr=True):
    """one expert per token (slot 0), weight 1"""
    keep=ex.lr
    if not lr:ex.lr=None
    ex.signs=scales(ex.art,L);Lk.set(e,ex,L);ex.lr=keep
    sel=torch.full((1,8),e,dtype=torch.int64,device=dev);rw=torch.zeros(1,8,device=dev).half();rw[0,0]=1
    return torch.stack([Lk(x[b:b+1],sel,rw)[0].clone() for b in range(x.shape[0])])
for src in ('text','mmself'):
    exs={E:load_expert(f'{root}/{src}/L{LY}/experts/E{E}.pt') for E in EIDS};H,I=exs[EIDS[0]].H,exs[EIDS[0]].I
    x=(torch.randn(4,H,device=dev)*0.05).half();xf=x.float()
    # (1)
    Lk=MoELayer(2,H,I,G=4)
    for E,ex in exs.items():
        for L in (2,4):
            y=run(Lk,ex,L,x);yd=dense_fwd(ex.art,L,xf);yd0=dense_fwd(strip_lr(ex.art),L,xf)
            ex.signs=scales(ex.art,L);yr=Expert.ref(ex,xf,L)
            a,b=rel(y,yd),rel(y,yr)
            ok&=a<3e-3 and b<3e-4
            msg=f'(1) {src} E{E} r {ex.rg}/{ex.rg}/{ex.rd} L{L}: vs decode_expert {a:.2e}  vs Expert.ref {b:.2e}  (|lr term|/|y| {(yd-yd0).norm()/yd.norm():.2e})'
            if ex.rg+ex.rd:
                eb=load_expert(None,art=boost(ex.art),verbose=False);yb=run(Lk,eb,L,x);ydb=dense_fwd(eb.art,L,xf)
                eb.signs=scales(eb.art,L);yrb=Expert.ref(eb,xf,L);ab,bb=rel(yb,ydb),rel(yb,yrb);ok&=ab<3e-3 and bb<3e-4
                msg+=f' | U x64: |lr|/|y| {(ydb-yd0).norm()/ydb.norm():.2f}, vs decode_expert {ab:.2e}, vs Expert.ref {bb:.2e}'
            print(msg,flush=True)
    # (2) mixed levels, one launch
    Lm=MoELayer(2,H,I,G=4);B=4
    sel=torch.zeros(B,8,dtype=torch.int64,device=dev);sel[:,1]=1;rw=torch.zeros(B,8,device=dev)
    rw[:,0]=torch.rand(B,device=dev)*0.5+0.25;rw[:,1]=1-rw[:,0];rw=rw.half()
    for lv in ((2,4),(4,2)):
        for E,l in zip(EIDS,lv):exs[E].signs=scales(exs[E].art,l)
        for e,E in enumerate(EIDS):Lm.set(e,exs[E],lv[e])
        yd=sum(rw[:,e:e+1].float()*dense_fwd(exs[E].art,lv[e],xf) for e,E in enumerate(EIDS))
        yr=sum(rw[:,e:e+1].float()*Expert.ref(exs[E],xf,lv[e]) for e,E in enumerate(EIDS))
        for cg,cd in (([1,8,3],[1,8,2]),([2,8,4,1],[2,8,4,1])):
            y=Lm(x,sel,rw,cfg_gu=cg,cfg_dn=cd).clone()
            a,b=rel(y,yd),rel(y,yr);ok&=a<3e-3 and b<3e-4
            print(f'(2) {src} mixed E{EIDS[0]}@L{lv[0]} E{EIDS[1]}@L{lv[1]} B{B} cfg {cg}/{cd}: vs decode_expert {a:.2e}  vs Expert.ref {b:.2e}',flush=True)
    clean=Lm.zws[:32*(I//128)*4].abs().max().item()==0 and Lm.wq.abs().sum().item()==0;ok&=clean;print(f'(2) workspace clean {clean}')
    del Lk,Lm
    # (3) TP8
    NS=8;E=169;art=exs[E].art;parts=[torch.load(f'{root}/{src}/L{LY}/tp{s}.pt',weights_only=False)[E] for s in range(NS)]
    lg,lu,ld=(art[p]['base']['lr'] for p in ('gate','up','down'));U4=lambda p:art[p]['p4']['lr']['U4']
    asm=(all(torch.equal(q['gate']['lrV'],lg['V']) and q['up'].get('lrV_from')=='gate' and torch.equal(q['down']['lrU2'],ld['U2'])
             and torch.equal(q['down']['lrU4'],U4('down')) for q in parts)
         and torch.equal(torch.cat([q['down']['lrV'] for q in parts],1),ld['V'])
         and all(torch.equal(torch.cat([q[p][k] for q in parts],1),t) for p,k,t in
                 (('gate','lrU2',lg['U2']),('up','lrU2',lu['U2']),('gate','lrU4',U4('gate')),('up','lrU4',U4('up')))))
    ok&=asm;print(f'(3) {src} E{E}: tp shard lr fields reassemble to the artifact lr (gate V replicated, down V column-sliced, down U replicated) {asm}')
    Is=I//NS;Ls=MoELayer(1,H,Is,G=4);cs=dict(cfg_gu=[1,8,2],cfg_dn=[1,8,1])
    sel=torch.zeros(1,8,dtype=torch.int64,device=dev);rw=torch.zeros(1,8,device=dev).half();rw[0,0]=1
    for L in (2,4):
        Dsum=0;hos=[];worst=0;SC=1024.
        for s,q in enumerate(parts):
            es=Expert(H,Is,seed=500+s,rk_gu=RK_OF[2.0],rk_dn=RK_OF[2.3125],var=True)
            gl,ul,dl=q['gate'],q['up'],q['down']
            es.set_lr(gl['lrV'],gl['lrU2'],ul['lrU2'],gl['lrU4'],ul['lrU4'],dl['lrV'],dl['lrU2'],dl['lrU4'])
            Ls.set(0,es,L);y=torch.stack([Ls(x[b:b+1],sel,rw,**cs)[0].clone() for b in range(4)])
            yr,ho=es.ref(xf,L,ret_ho=True);worst=max(worst,rel(y,yr));hos.append(ho)
            es.set_lr(gl['lrV'],gl['lrU2'],ul['lrU2'],gl['lrU4'],ul['lrU4'],dl['lrV'],dl['lrU2']*SC,dl['lrU4']*SC)  # down U x1024
            Ls.set(0,es,L);y=torch.stack([Ls(x[b:b+1],sel,rw,**cs)[0].clone() for b in range(4)])
            worst=max(worst,rel(y,es.ref(xf,L)))
            es.set_lr(gl['lrV'],gl['lrU2'],ul['lrU2'],gl['lrU4'],ul['lrU4'],dl['lrV'][:0],dl['lrU2'][:0],dl['lrU4'][:0])  # down lr off
            Ls.set(0,es,L);y0=torch.stack([Ls(x[b:b+1],sel,rw,**cs)[0].clone() for b in range(4)])
            Uf=SC*(ld['U2'].float().to(dev)+(U4('down').float().to(dev) if L>=4 else 0))[0]
            Dsum=Dsum+((y-y0)@Uf)/(Uf@Uf)      # this rank's partial z (rank-1 down plane)
        assert ld['V'].shape[0]==1
        full=(torch.cat(hos,1)@ld['V'].float().to(dev).T)[:,0]
        c=rel(Dsum,full);ok&=worst<3e-4 and c<3e-4
        print(f'(3) {src} L{L}: per-rank shard kernel (real shard lr, synthetic planes) vs Expert.ref max {worst:.2e};  '
              f'sum_s kernel partial z_s vs ho_full @ V_d_full^T {c:.2e}',flush=True)
    del Ls,exs
print('PASS' if ok else 'FAIL')
