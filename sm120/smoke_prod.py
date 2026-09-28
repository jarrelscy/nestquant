"""Production-format smoke test: encode a few real GLM-5.3 experts with quickfit/fit_prod.py (4.1263 bpw layout, sign
base variant), write them exactly like threads/12 nq_layer.py (L{L}/tp{s}.pt + manifest.json), then read them back the
way a loader would and run them through the sm120 kernel:
 (1) manifest sizes == threads/13 INTEGRATION.md per expert-shard table (4.1263 row)
 (2) nq_layer.assemble(shard files) decodes == the in-memory artifact, levels 2/4
 (3) per TP group (TP8: 1 shard, TP4: shards 2r,2r+1, TP1: all 8) built from the shard files only:
     nq_decode of the group == rows/cols slice of the full-expert decode (bitwise);
     kernel decode (NQ_WDUMP) == nq_decode of the group, bitwise, levels 2/4, all 3 projections;
 (4) kernel forward, sum of per-rank partial outputs vs x through the full dense expert (TP8, TP4, TP1).
The low-rank plane (f128e41) is included everywhere: fit_prod gives each expert a deterministic rank 0..4 per gate|up /
down, the group artifacts rebuild base.lr / p4.lr from the shard keys like nq_layer.assemble, the kernel adds the lr
term (table [14..17]) and D.decode_expert (reference) adds lr_term to the dense weights.
usage: smoke_prod.py [L] [E0:E1]   (default L3 E0:5 covers r_gu = 0 and r_dn = 0)"""
import os,sys,types,json,torch;torch.cuda.set_per_process_memory_fraction(16/96)
HERE=os.path.dirname(os.path.abspath(__file__));sys.path[:0]=[HERE,HERE+'/quickfit']
import fit_prod as FP
NE,D,NL=FP.NE,FP.D,FP.NL
import moe;from moe import *
from build import get
dev='cuda';torch.backends.cuda.matmul.allow_tf32=False
L=int(sys.argv[1]) if len(sys.argv)>1 else 3
e0,e1=map(int,(sys.argv[2] if len(sys.argv)>2 else '0:5').split(':'))
ROOT='/data/Jarrel/nq-glm53-prod/smoke';ok=True
def chk(c,msg):
    global ok;ok&=bool(c);print(('  ok  ' if c else '  FAIL')+' '+msg,flush=True)
# ---------------------------------------------------------------- encode + write
arts={}
for E in range(e0,e1):
    arts[E],_,_=FP.fit(L,E);print(f'encoded L{L} E{E} rate {arts[E]["meta"]["rate"]:.4f}',flush=True)
man=FP.finalize(f'{ROOT}/L{L}',L,arts)
# ---------------------------------------------------------------- (1) sizes
pb=man['packed_bytes_per_expert_per_shard'];print('(1) bytes per expert-shard',json.dumps(pb))
chk(pb['gate']['base']+pb['up']['base']==786432 and pb['down']['base']==393216,'base 786,432 gu / 393,216 down')
chk(pb['gate']['p4']+pb['up']['p4']==786432 and pb['down']['p4']==454656,'P4 786,432 gu / 454,656 down (4.1263 row)')
chk(pb['gate']['word']+pb['up']['word']==3072 and pb['down']['word']==1536,'block words u16 (d4 u32 in the kernel = 6,144 / 3,072)')
chk(pb['gate']['var']+pb['up']['var']==1536 and pb['down']['var']==768,'sign variant 1 bit/ring = 1,536 gu / 768 down (uint8 per unit)')
for E in arts:
    r=man['per_expert'][str(E)]['lr_rank'] if str(E) in man['per_expert'] else man['per_expert'][E]['lr_rank']
    chk(r=={'gate':FP.lr_rank(L,E,'gu'),'up':FP.lr_rank(L,E,'gu'),'down':FP.lr_rank(L,E,'dn')},f'E{E} manifest lr_rank {r}')
chk(man['config']['base_var']=='sign' and man['config']['res_K']=={'gate':2.0,'up':2.0,'down':2.3125},'config base_var sign, res_K 2/2/2.3125')
# ---------------------------------------------------------------- (2) assemble
parts=[torch.load(f'{ROOT}/L{L}/tp{s}.pt',weights_only=False) for s in range(8)]
for E in arts:
    re=NL.assemble(ROOT,L,E)
    chk(all(all(torch.equal(x,y) for x,y in zip(D.decode_expert(arts[E],lv),D.decode_expert(re,lv))) for lv in (2,4)),f'(2) E{E} assemble(tp*.pt) decode == artifact decode, levels 2/4')
# ---------------------------------------------------------------- (3)/(4) TP groups from shard files
import nqload as NQ
group_art=lambda E,ss:NQ.group_art(parts,man,E,ss)
kernel_expert=lambda art:NQ.kernel_expert(art,dev,want_Q=True)
Md=get(['NQ_WDUMP']);Mn=get([])
H=6144;sel=torch.zeros(1,8,dtype=torch.int64,device=dev);rw=torch.zeros(1,8,device=dev).half();rw[0,0]=1
xs=(torch.randn(4,H,device=dev)*0.05).half()
for E in arts:
    full={lv:D.decode_expert(arts[E],lv,dev) for lv in (2,4)}
    for tp in (8,4,1):
        g=8//tp;ysum={2:0,4:0};nbit=0;nall=0;nsl=0
        for r in range(tp):
            ss=list(range(r*g,(r+1)*g));ga=group_art(E,ss);I=256*g;sl=slice(r*I,(r+1)*I)
            for lv in (2,4):                              # group decode == slice of the full decode
                Wg,Wu,Wd=D.decode_expert(ga,lv,dev);F_=full[lv]
                nsl+=not(torch.equal(Wg,F_[0][sl]) and torch.equal(Wu,F_[1][sl]) and torch.equal(Wd,F_[2][:,sl]))
            ex,scales,(Qg,Qu,Qd)=kernel_expert(ga)
            Lr=MoELayer(1,H,I,G=4,mod=Md);Lf=MoELayer(1,H,I,G=4,mod=Mn)
            for lv in (2,4):
                ex.signs=scales(lv);Lr.set(0,ex,lv)
                wg=torch.full((2*I,H),float('nan'),device=dev).half();wd=torch.full((H,I),float('nan'),device=dev).half()
                Md.set_wdump(wg.data_ptr(),wd.data_ptr(),0);Lr(xs[:1],sel,rw);torch.cuda.synchronize();Md.set_wdump(0,0,-1)
                for w,q in ((wg[:I],Qg[lv]),(wg[I:],Qu[lv]),(wd,Qd[lv])):
                    nall+=1;nbit+=not torch.equal(w.view(torch.int16),q.view(torch.int16))
                Lf.set(0,ex,lv);ysum[lv]=ysum[lv]+torch.stack([Lf(xs[b:b+1],sel,rw)[0].float().clone() for b in range(4)])
            del Lr,Lf,wg,wd;torch.cuda.empty_cache()
        chk(nsl==0,f'(3) E{E} TP{tp}: group decode from shard files == full decode slice ({tp} ranks x 2 levels)')
        chk(nbit==0,f'(3) E{E} TP{tp}: kernel decode == nq_decode bitwise, {nall} projections ({nbit} mismatching)')
        for lv in (2,4):
            Wg,Wu,Wd=full[lv];xf=xs.float();yr=(torch.nn.functional.silu(xf@Wg.T)*(xf@Wu.T))@Wd.T
            rel=((ysum[lv]-yr).norm()/yr.norm()).item();chk(rel<3e-3,f'(4) E{E} TP{tp} level {lv}: sum of rank partials vs dense expert, rel err {rel:.2e}')
print('SMOKE PASS' if ok else 'SMOKE FAIL')
