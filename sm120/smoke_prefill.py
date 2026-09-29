"""Prefill path (MoELayer.prefill: decode each routed expert once + grouped fp16 GEMMs, fp32 out) on real serving layers
(repack dir: res/rank{r}/L{L}.pt + rank{r}.bin records), half the experts at level 4 (record in a pool slot, p4rec.row),
the rest level 2, random distinct top-8 routing (some rw = 0):
 (a) pf_decode weights == the decode kernel's own decoded weights (NQ_WDUMP build), bitwise, level 2 + level 4 experts;
 (b) prefill output vs an fp32 torch reference (same decoded weights, rotations / sv / lr / SwiGLU / in_had_down / rw in fp32);
 (c) prefill output vs the T <= 8 decode-kernel slice loop (what forward() did before);
 (d) routing hits exported by prefill == hits of the slice loop;
 (e) layer time: slice loop vs prefill.
usage: smoke_prefill.py [REPACK] [layers=3,4,5,6,10] [Ts=9,64,512,4096,16384] [rank=0]"""
import os,sys,json,time,random,torch
HERE=os.path.dirname(os.path.abspath(__file__));sys.path[:0]=[HERE,HERE+'/../streaming']
import moe;from moe import MoELayer,entry,H128
from build import get
import resident as RS,p4rec as PR,stream_engine as SE
rp=sys.argv[1] if len(sys.argv)>1 else '/home/jarrelscy/nq-p4rec/hf'
Ls=[int(v) for v in (sys.argv[2] if len(sys.argv)>2 else '3,4,5,6,10').split(',')]
Ts=[int(v) for v in (sys.argv[3] if len(sys.argv)>3 else '9,64,512,4096,16384').split(',')]
rank=int(sys.argv[4]) if len(sys.argv)>4 else 0
dev='cuda';torch.backends.cuda.matmul.allow_tf32=False
Mn=get([]);Md=get(['NQ_WDUMP']);rf=SE.RankFile(rp,rank);rb=rf.rb;NE=256;TK=8
ok=True
def chk(c,msg):
    global ok;ok&=bool(c);print(('  ok  ' if c else '  FAIL')+' '+msg,flush=True)
for L in Ls:
    ex,H,I=RS.load(f'{rp}/res/rank{rank}/L{L}.pt',dev);rng=random.Random(L);hw=moe.had_dn(ex[0])
    l4=set(rng.sample(range(NE),NE//2));pool=torch.empty(len(l4),rb,dtype=torch.uint8,device=dev);slot={}
    fd=os.open(rf.path,os.O_RDONLY);buf=torch.empty(rb,dtype=torch.uint8).pin_memory()
    for i,E in enumerate(sorted(l4)):
        assert os.preadv(fd,[memoryview(buf.numpy())],rf.rec(L,E)*rb)==rb;pool[i].copy_(buf);slot[E]=i
    os.close(fd)
    Ms={}
    for nm,mod in (('n',Mn),('d',Md)):
        M=MoELayer(NE,H,I,Bmax=8,mod=mod)
        for E,x in ex.items():M.table[E].copy_(PR.row(x,rf.lay,pool[slot[E]].data_ptr(),entry) if E in l4 else entry(x,2))
        Ms[nm]=M
    M=Ms['n'];lv=[4 if E in l4 else 2 for E in range(NE)]
    print(f'L{L} rank {rank}: H {H} I {I} in_had_down {hw}, {len(l4)} experts at level 4, lr gu/dn {sum(x.rg>0 for x in ex.values())}/{sum(x.rd>0 for x in ex.values())}',flush=True)
    # ------------------------------------------------ (a) decoded weights, bitwise vs the decode kernel's WDUMP
    lrE=[E for E in range(NE) if ex[E].rg>0 and ex[E].rd>0]
    pick=[E for E in range(NE) if E not in l4][:2]+[E for E in l4][:2]+[E for E in lrE if E in l4][:1]+[E for E in lrE if E not in l4][:1]
    Wgu=torch.empty(2*2*I*H,dtype=torch.float16,device=dev);Wdn=torch.empty(2*H*I,dtype=torch.float16,device=dev)
    for E in pick:
        wg=torch.zeros(2*I,H,dtype=torch.float16,device=dev);wd=torch.zeros(H,I,dtype=torch.float16,device=dev)
        x1=(torch.randn(1,H,device=dev)*0.05).half();sel=torch.tensor([[E]+[e for e in range(NE) if e!=E][:TK-1]],device=dev)
        Md.set_wdump(wg.data_ptr(),wd.data_ptr(),E);Ms['d'](x1,sel,torch.full((1,TK),0.125,device=dev).half());torch.cuda.synchronize();Md.set_wdump(0,0,-1)
        Mn.pf_decode(M.table,torch.tensor([E,E],dtype=torch.int32,device=dev),Wgu,Wdn,H,I,0,0)
        g1=Wgu[:2*I*H].view(2*I,H);d1=Wdn[:H*I].view(H,I)
        chk(torch.equal(g1,wg) and torch.equal(d1,wd) and torch.equal(Wgu[2*I*H:].view(2*I,H),wg),f'(a) E{E} level {lv[E]} rg/rd {ex[E].rg}/{ex[E].rd}: pf_decode == kernel WDUMP (gu, dn)')
    # decoded weights of every expert (for the fp32 reference)
    Wall={}
    for e0 in range(0,NE,16):
        es=torch.arange(e0,e0+16,dtype=torch.int32,device=dev);Wg=torch.empty(16*2*I*H,dtype=torch.float16,device=dev);Wd=torch.empty(16*H*I,dtype=torch.float16,device=dev)
        Mn.pf_decode(M.table,es,Wg,Wd,H,I,0,0)
        for j in range(16):Wall[e0+j]=(Wg.view(16,2*I,H)[j],Wd.view(16,H,I)[j])
    Hm=H128(dev);Hd=H128(dev,hw);wht=lambda v,Mh=Hm:(v.reshape(*v.shape[:-1],-1,Mh.shape[0])@Mh).reshape(v.shape)
    sg4=rf.lay['seg']['lr4'][0]
    def ref_expert(E,xf):
        x=ex[E];s=(x.sc[lv[E]]).float();su,svg,svu,sud,svo,suu=s[:H],s[H:H+I],s[H+I:H+2*I],s[H+2*I:H+3*I],s[H+3*I:2*H+3*I],s[2*H+3*I:]
        Wg,Wd=[w.float() for w in Wall[E]]
        a_g=wht(xf*su).half().float()@Wg[:I].T;a_u=wht(xf*suu).half().float()@Wg[I:].T
        g=wht(a_g)*svg;u=wht(a_u)*svu;rg,rd=x.rg,x.rd
        if rg+rd:
            lr=x.lr.float();o=0
            Vg=lr[:rg*H].view(rg,H);o=rg*H;U2g=lr[o:o+rg*I].view(rg,I);o+=rg*I;U2u=lr[o:o+rg*I].view(rg,I);o+=rg*I
            Vd=lr[o:o+rd*I].view(rd,I);o+=rd*I;U2d=lr[o:o+rd*H].view(rd,H)
            if lv[E]==4:
                l4v=pool[slot[E]][sg4:sg4+2*(2*rg*I+rd*H)].view(torch.float16).float()
                U4g=l4v[:rg*I].view(rg,I);U4u=l4v[rg*I:2*rg*I].view(rg,I);U4d=l4v[2*rg*I:].view(rd,H)
            else:U4g=torch.zeros_like(U2g);U4u=torch.zeros_like(U2u);U4d=torch.zeros_like(U2d)
            z=xf@Vg.T;g=g+z@U2g+z@U4g;u=u+z@U2u+z@U4u
        sw=torch.nn.functional.silu(g)*u
        h=wht(sw*sud,Hd).half().float();y=wht(h@Wd.T)*svo
        if rd:z=sw@Vd.T;y=y+z@U2d+z@U4d
        return y
    for T in Ts:
        g=torch.Generator(device=dev).manual_seed(1000*L+T)
        pop=torch.randn(NE,device=dev,generator=g)*1.5   # skewed expert popularity, distinct top-8 per token (as the router)
        sel=torch.topk(pop[None]+torch.rand(T,NE,device=dev,generator=g).log().neg().log().neg(),TK,1).indices
        rw=torch.softmax(torch.randn(T,TK,device=dev,generator=g),1).half();rw[::7,3]=0
        x=(torch.randn(T,H,device=dev,generator=g)*0.05).half()
        hp=torch.zeros(NE,dtype=torch.int32).pin_memory();hs=torch.zeros(NE,dtype=torch.int32).pin_memory()
        M.hits_ptr=hs.data_ptr();ys=torch.empty(T,H,dtype=torch.float32,device=dev)
        for i in range(0,T,8):j=min(T,i+8);M(x[i:j],sel[i:j],rw[i:j],out=ys[i:j],cfg_gu=[1,8,6],cfg_dn=[1,8,4])
        M.hits_ptr=hp.data_ptr();yp=M.prefill(x,sel,rw);torch.cuda.synchronize();M.hits_ptr=0
        ref=torch.zeros(T,H,device=dev);xf=x.float();fl=sel.flatten();tokk=torch.arange(T*TK,device=dev)//TK
        for E in fl.unique().tolist():
            idx=(fl==E).nonzero().flatten();t=tokk[idx];w=rw.flatten()[idx].float()
            for c in range(0,len(t),4096):ref.index_add_(0,t[c:c+4096],w[c:c+4096,None]*ref_expert(E,xf[t[c:c+4096]]))
        rn=ref.norm();e_p=((yp-ref).norm()/rn).item();e_s=((ys-ref).norm()/rn).item();e_ps=((yp-ys).norm()/ys.norm()).item()
        chk(e_p<3e-3 and e_ps<3e-3 and torch.isfinite(yp).all(),f'(b,c) T{T}: rel err prefill vs fp32 ref {e_p:.2e} (slice loop vs ref {e_s:.2e}), prefill vs slice loop {e_ps:.2e}; '
            f'{int(sum(lv[e]==4 for e in fl.tolist()))}/{T*TK} picks level 4')
        chk(torch.equal(hp,hs),f'(d) T{T}: prefill hits == slice-loop hits (sum {int(hp.sum())})')
        # (e) timing
        def tm(f,n=3):
            f();torch.cuda.synchronize();a=torch.cuda.Event(True);b=torch.cuda.Event(True);a.record()
            for _ in range(n):f()
            b.record();b.synchronize();return a.elapsed_time(b)/n*1e3
        def sl():
            for i in range(0,T,8):j=min(T,i+8);M(x[i:j],sel[i:j],rw[i:j],out=ys[i:j],cfg_gu=[1,8,6],cfg_dn=[1,8,4])
        ts=tm(sl,1 if T>=4096 else 3);tp=tm(lambda:M.prefill(x,sel,rw,out=yp))
        print(f'  (e) T{T}: layer slice loop {ts:.0f} us, prefill {tp:.0f} us ({ts/tp:.1f}x), prefill {tp/T:.2f} us/token',flush=True)
        del ref,yp,ys;torch.cuda.empty_cache()
    del Wall,pool,Ms,M,ex;torch.cuda.empty_cache()
print('PREFILL SMOKE PASS' if ok else 'PREFILL SMOKE FAIL')
