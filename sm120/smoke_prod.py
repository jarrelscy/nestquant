"""Production-format smoke test: encode a few real GLM-5.3 experts with quickfit/fit_prod.py (4.1263 bpw layout, sign
base variant), write them exactly like threads/12 nq_layer.py (L{L}/tp{s}.pt + manifest.json), then read them back the
way a loader would and run them through the sm120 kernel:
 (1) manifest sizes == threads/13 INTEGRATION.md per expert-shard table (4.1263 row)
 (2) nq_layer.assemble(shard files) decodes == the in-memory artifact, levels 2/4
 (3) per TP group (TP8: 1 shard, TP4: shards 2r,2r+1, TP1: all 8) built from the shard files only:
     nq_decode of the group == rows/cols slice of the full-expert decode (bitwise);
     kernel decode (NQ_WDUMP) == nq_decode of the group, bitwise, levels 2/4, all 3 projections;
 (4) kernel forward, sum of per-rank partial outputs vs x through the full dense expert (TP8, TP4, TP1).
usage: smoke_prod.py [L] [E0:E1]"""
import os,sys,types,json,torch;torch.cuda.set_per_process_memory_fraction(16/96)
HERE=os.path.dirname(os.path.abspath(__file__));sys.path[:0]=[HERE,HERE+'/quickfit']
import fit_prod as FP
NE,D,NL=FP.NE,FP.D,FP.NL
import moe;from moe import *
from build import get
dev='cuda';torch.backends.cuda.matmul.allow_tf32=False
L=int(sys.argv[1]) if len(sys.argv)>1 else 3
e0,e1=map(int,(sys.argv[2] if len(sys.argv)>2 else '0:3').split(':'))
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
chk(man['config']['base_var']=='sign' and man['config']['res_K']=={'gate':2.0,'up':2.0,'down':2.3125},'config base_var sign, res_K 2/2/2.3125')
# ---------------------------------------------------------------- (2) assemble
parts=[torch.load(f'{ROOT}/L{L}/tp{s}.pt',weights_only=False) for s in range(8)]
for E in arts:
    re=NL.assemble(ROOT,L,E)
    chk(all(all(torch.equal(x,y) for x,y in zip(D.decode_expert(arts[E],lv),D.decode_expert(re,lv))) for lv in (2,4)),f'(2) E{E} assemble(tp*.pt) decode == artifact decode, levels 2/4')
# ---------------------------------------------------------------- (3)/(4) TP groups from shard files
def group_art(E,ss):
    """pseudo-artifact of the contiguous shard group ss of expert E (a self-contained expert of I = 256*len(ss))."""
    art={}
    for pn in NE.PROJ:
        m=dict(man['proj_meta'][pn]);ps=[parts[s][E][pn] for s in ss];g=len(ss)
        if pn=='down':m.update(k=m['k']*g//8,tk=m['tk']*g//8);suh2=torch.cat([p['suh2'] for p in ps]);svh2=ps[0]['svh2'];suh4=torch.cat([p['suh4'] for p in ps]);svh4=ps[0]['svh4']
        else:m.update(n=m['n']*g//8,tn=m['tn']*g//8);suh2=ps[0]['suh2'];svh2=torch.cat([p['svh2'] for p in ps]);suh4=ps[0]['suh4'];svh4=torch.cat([p['svh4'] for p in ps])
        art[pn]=dict(base=dict(shards=[p['base'] for p in ps],var=[p['var'] for p in ps],suh=suh2,svh=svh2),
                     p4=dict(shards=[p['p4'] for p in ps],word=[p['word'] for p in ps],suh=suh4,svh=svh4),meta=m)
    return art
def repack(P):
    """(threads/13 verify_t12.repack) T12 planes -> kernel Proj fields + rotated nq_decode Q2/Q4 [N,K] fp16."""
    m=P['meta'];tk,tn=m['tk'],m['tn'];U=tk*tn
    rule=m['res_rule'];assert rule['kind']=='uniform',rule
    K=float(rule['K']);rk=RK_OF[K];KA,MASK=RKP[rk];assert D.PATTERNS[K]==(KA,MASK)
    rl=D.ring_levels(P,dev);order=rl['order']
    Q={lv:D.to_matrix(rl[f'Q{lv}'],order,tk,tn,m['k'],m['n'],dev).T.contiguous().half() for lv in (2,4)}
    rec=torch.empty(U,dtype=torch.long,device=dev);a,c=order//tn,order%tn;rec[:]=c*tk+a;inv=torch.argsort(rec)
    def lane_words(raw,bits):
        nb=8*4*bits//8;st=raw.to(dev).view(U,nb)[inv]
        b=((st.long()[...,None]>>torch.arange(8,device=dev))&1).view(U,8,4,bits).reshape(U*32,bits)
        nw=(bits+31)//32;b=torch.nn.functional.pad(b,(0,nw*32-bits)).view(U*32,nw,32)
        return (b<<torch.arange(32,device=dev)).sum(-1)
    wb=lane_words(D._cat(P['base']['shards'],'cpu'),128);wr=lane_words(D._cat(P['p4']['shards'],'cpu'),rbits(rk))
    p=types.SimpleNamespace(N=m['n'],K=m['k'],rk=rk,z=proj_sizes(m['n'],m['k'],None,rk),flags=None,fl=None)
    p.base=torch.from_numpy(wb.reshape(-1).cpu().numpy().astype('uint32').view('int32')).to(dev)
    p.p4w=wr;p.p4=pack_words(wr.cpu(),rbits(rk)).to(dev)
    bw=D._cat(P['p4']['word'],'cpu').long().to(dev)[inv];p.Mb=bw&255;p.Nn=(bw>>8)&255;p.d4=(p.Mb|(p.Nn<<8)).to(torch.int32)
    assert m['base_var']=='sign'
    v=D._cat(P['base']['var'],'cpu').long().to(dev).view(U,8)[inv];p.var=((v&1)<<torch.arange(8,device=dev)).sum(1).to(torch.uint8)
    return p,Q
def cat_proj(a,b):
    p=types.SimpleNamespace(N=a.N+b.N,K=a.K,rk=a.rk,z=proj_sizes(a.N+b.N,a.K,None,a.rk),flags=None,fl=None)
    p.base=torch.cat([a.base,b.base]);p.p4w=torch.cat([a.p4w,b.p4w]);p.p4=pack_words(p.p4w.cpu(),rbits(p.rk)).to(dev)
    p.Mb=torch.cat([a.Mb,b.Mb]);p.Nn=torch.cat([a.Nn,b.Nn]);p.d4=torch.cat([a.d4,b.d4]);p.var=torch.cat([a.var,b.var]);return p
def kernel_expert(art):
    g,Qg=repack(art['gate']);u,Qu=repack(art['up']);d,Qd=repack(art['down'])
    ex=types.SimpleNamespace(gu=cat_proj(g,u),dn=d,H=d.N,I=d.K)
    def scales(lv):
        pl=D.SCALE_PLANE[lv];s=lambda n,k:art[n][pl][k].half().to(dev)
        return torch.cat([s('gate','suh'),s('gate','svh'),s('up','svh'),s('down','suh'),s('down','svh'),s('up','suh')])
    return ex,scales,(Qg,Qu,Qd)
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
