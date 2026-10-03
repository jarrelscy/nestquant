"""Bit-exact kernel check for nq-res-v2 (b1.75/4, threads/35): base K code 1 (1.75, table [19]) and residual code 9 (2.5625).
 (a) synthetic experts, bk {0,1} x rk {0,1,2,3,4,7,8,9}, dense + mask mode, base-variant signs: kernel NQ_WDUMP decode == moe.dense_W
     (bitwise fp16, +0/-0 equal), levels 2 and 4, G = 4, at I = 512 (TP4) and I = 1024 (TP2, 2x Spark)
 (b) real test vectors (/tmp/nestquant/35-nq15/testvec, sm120/NQ_RES_V2.md section 4): res row + record bytes put into a table row the
     way the server does (resident planes + p4rec.row on a slot), kernel decode == the vector's W at levels 2 and 4
Both the decode kernel (NQ_WDUMP) and the prefill decode (pf_decode, unit_dec) are checked.
Build: NQ_DEFS=NQ_WDUMP,NQ_RK_CODES=0x3ff,NQ_BK_CODES=0x3 (TORCH_CUDA_ARCH_LIST may be 8.0: decode uses sm_80 instructions only).
Usage: verify_b175.py [testvec_dir]"""
import os,sys,glob,types,torch
torch.cuda.set_per_process_memory_fraction(float(os.environ.get('NQ_MEMFRAC','0.1')))
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','streaming'))
import moe;from moe import *
import p4rec as PR,resident as RS
from build import get
TV=sys.argv[1] if len(sys.argv)>1 else '/tmp/nestquant/35-nq15/testvec'
Md=get(['NQ_WDUMP','NQ_RK_CODES=0x3ff','NQ_BK_CODES=0x3'])
print('rk_codes',Md.rk_codes(),'bk_codes',Md.bk_codes(),flush=True)
assert Md.bk_codes()[0]&2 and Md.bk_codes()[1]&2 and Md.rk_codes()[1]>>9&1
H=6144
def cmp(w,r):
    bw=w.view(torch.int16)!=r.view(torch.int16);z=bw&(w==0)&(r==0);m=bw&~z
    return int(m.sum()),(m.nonzero()[:3].tolist() if m.any() else [])
def dump(L,e,I,x,sel,rw):
    wg=torch.full((2*I,H),float('nan'),device='cuda').half();wd=torch.full((H,I),float('nan'),device='cuda').half()
    Md.set_wdump(wg.data_ptr(),wd.data_ptr(),e);L(x,sel,rw);torch.cuda.synchronize();Md.set_wdump(0,0,-1)
    return wg,wd
def pfdump(L,e,I):
    """prefill path decode (nq_pf_decode / unit_dec) of expert e from the live table row"""
    wg=torch.full((2*I*H,),float('nan'),device='cuda').half();wd=torch.full((H*I,),float('nan'),device='cuda').half()
    Md.pf_decode(L.table,torch.tensor([e],dtype=torch.int32,device='cuda'),wg,wd,H,I,L.nm_gu,L.nm_dn);torch.cuda.synchronize()
    return wg.view(2*I,H),wd.view(H,I)
tot=bad=0
# ---- (a) synthetic
cases=[(1,1,3,9),(1,1,0,9),(1,1,9,3),(1,0,3,7),(0,1,8,9),(0,0,3,9),(1,1,1,2),(1,1,7,4)]   # (bk_gu, bk_dn, rk_gu, rk_dn)
masked={1,5}
for I in (512,1024):
    nmg,nmd=H//128//2,I//128//2
    ex=[Expert(H,I,seed=900+i,nm_gu=nmg if i in masked else None,nm_dn=nmd if i in masked else None,rk_gu=c[2],rk_dn=c[3],
               var=i%2==0,bk_gu=c[0],bk_dn=c[1]) for i,c in enumerate(cases)]
    sel=torch.arange(8,device='cuda')[None];rw=torch.full((1,8),0.125,device='cuda').half();x=(torch.randn(1,H,device='cuda')*0.05).half()
    L=MoELayer(8,H,I,nmg,nmd,G=4,mod=Md)
    for lvl in (2,4):
        for e in range(8):L.set(e,ex[e],lvl)
        for e in range(8):
            wg,wd=dump(L,e,I,x,sel,rw);pg,pd=pfdump(L,e,I)
            for nm,pj,w in (('gu',ex[e].gu,wg),('dn',ex[e].dn,wd),('pf.gu',ex[e].gu,pg),('pf.dn',ex[e].dn,pd)):
                n,f=cmp(w,dense_W(pj,lvl,4,torch.float16));tot+=1;bad+=n>0
                if n:print(f'  MISMATCH I{I} L{lvl} e{e} {nm} bk {pj.bk} rk {pj.rk} mask {pj.fl is not None} n {n} first {f}',flush=True)
        print(f'(a) I={I} level {lvl}: {tot} projections so far, {bad} mismatching',flush=True)
    del L,ex
# ---- (b) real vectors
for p in sorted(glob.glob(f'{TV}/L*_E*_r*.pt')):
    tv=torch.load(p,weights_only=False);m=tv['meta'];I=m['I'];r=tv['res']
    g={k:v.cuda() for k,v in r.items()};dummy=torch.zeros(1,dtype=torch.float16,device='cuda')
    P=lambda k,rk,bk:types.SimpleNamespace(base=g[k+'_base'],var=g[k+'_var'],p4=None,d4=None,flags=None,fl=None,rk=rk,bk=bk)
    has=m['rg']+m['rd']>0
    xe=types.SimpleNamespace(gu=P('gu',m['rk_gu'],m['bk_gu']),dn=P('dn',m['rk_dn'],m['bk_dn']),H=H,I=I,rg=m['rg'],rd=m['rd'],
                             sc={2:g['sc2'],4:g['sc4']},lr=g['lr'][:RS.lr_len(H,I,m['rg'],m['rd'])] if has else None,lr4=dummy if has else None,
                             had_dn=m['in_had_down'])
    xe.signs=xe.sc[2]
    slot=tv['record'].cuda();lay=dict(seg={k:tuple(v) for k,v in m['seg'].items()},rec_bytes=m['rec_bytes'])
    L=MoELayer(8,H,I,0,0,G=4,mod=Md)
    for lvl in (2,4):
        L.set(0,xe,2)                                     # runs the bk/rk compiled-code asserts
        if lvl==4:
            assert Md.rk_codes()[0]>>m['rk_gu']&1 and Md.rk_codes()[1]>>m['rk_dn']&1
            L.table[0].copy_(PR.row(xe,lay,slot.data_ptr(),entry).to('cuda'))
        sel=torch.zeros(1,8,dtype=torch.int64,device='cuda');rw=torch.zeros(1,8,device='cuda').half();rw[0,0]=1
        x=(torch.randn(1,H,device='cuda')*0.05).half()
        wg,wd=dump(L,0,I,x,sel,rw);pg,pd=pfdump(L,0,I)
        for nm,w in (('gu',wg),('dn',wd),('pf.gu',pg),('pf.dn',pd)):
            n,f=cmp(w,tv['W'][f'{nm[-2:]}{lvl}'].cuda());tot+=1;bad+=n>0
            if n:print(f'  MISMATCH {os.path.basename(p)} L{lvl} {nm} n {n} first {f}',flush=True)
    print(f'(b) {os.path.basename(p)} bk {m["bk_gu"]}/{m["bk_dn"]} rk {m["rk_gu"]}/{m["rk_dn"]} had_dn {m["in_had_down"]} I {I}: '
          f'{tot} projections so far, {bad} mismatching',flush=True)
    del L
print(f'verify_b175: {tot} projections, {bad} mismatching')
sys.exit(1 if bad else 0)
