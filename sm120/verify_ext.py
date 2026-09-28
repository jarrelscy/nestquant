"""Bit-exact verification of the nqdec decoder port, extended: residual codes 0..8 (8 = K 1.875 (1,0xFEFE), 6 = K 1.9375 (1,0xFFFE) gate/up,
7 = K 2.3125 (2,0x9248) down: the 4.0846 bpw production rate) and base-variant sign planes (T13 d9f844e: uint8 per unit, bit r = ring r, G = 4 only).
 (a) torch reference (moe.lane_vals / dense_W) == ref15_spec.decode_unit on sampled units, every residual code, G=4
 (b) kernel decoded weights (NQ_WDUMP build) == torch reference, bitwise, for levels 2/4, codes 0..5, dense + mask
     mode, G=4 and G=2; experts built with exhaustive (Mb, N) coverage (every valid block word appears)."""
import sys,torch,numpy as np;torch.cuda.set_per_process_memory_fraction(12/80)
import moe;from moe import *;import ref15_spec as R
from build import get
torch.manual_seed(0)
H=6144;I=int(sys.argv[1]) if len(sys.argv)>1 else 2048
# ---- (a) spec cross-check
nbad=0;nun=0
for rk in range(9):
    KA,MASK=RKP[rk];gen=torch.Generator().manual_seed(rk)
    p=Proj(64,512,gen,None,rk,('exh',rk*1000),var=rk%2==1)                    # 4 strips x 4 chunks
    lv2=lane_vals(p,2,4);lv4=lane_vals(p,4,4);W4=dense_W(p,4,4)
    S_,C_=p.z['S'],p.z['C'];wb=(p.base.long()&0xFFFFFFFF).view(-1,4).numpy()
    for s in range(S_):
        for c in range(C_):
            rec=(s*C_+c)*32;bs=R.rings_from_lane_words(wb[rec:rec+32],128)
            rs=R.rings_from_lane_words(p.p4w[rec:rec+32].numpy(),rbits(rk))
            Mb,N=int(p.Mb[s*C_+c]),int(p.Nn[s*C_+c])
            Q2,Q4=R.decode_unit(bs,rs,Mb,N,Kb=2,Kr=(KA,MASK))
            if p.var is not None:a=np.where(int(p.var[s*C_+c])>>np.arange(8)&1,-1.0,1.0)[:,None];Q2,Q4=a*Q2,a*Q4
            t2=lv2[s,c].reshape(8,256).double().numpy();t4=lv4[s,c].reshape(8,256).double().numpy()
            u=R.to_unit(Q4).T;d=W4[s*16:(s+1)*16,c*128:(c+1)*128].double().numpy()
            ok=np.array_equal(Q2,t2) and np.array_equal(Q4,t4) and np.array_equal(u,d);nbad+=not ok;nun+=1
print(f'(a) torch ref vs ref15_spec: {nun} units x 2 levels, codes 0..8 (odd codes signed), mismatching units {nbad}',flush=True)
assert nbad==0
# ---- (b) kernel vs torch ref
Md=get(['NQ_WDUMP','NQ_RK_CODES=0x1ff'])
nmg,nmd=H//128//2,I//128//2
ngu=(2*I//16)*(H//128);ndn=(H//16)*(I//128);nL=len(moe.all_MbN())
rks=[(0,8),(1,2),(2,1),(3,4),(4,5),(5,3),(6,7),(8,6)]   # every code 0..8 (8 = 1.875 both sides; 7 = 0x9248 down)
masked={6,7}
ex=[Expert(H,I,seed=500+i,nm_gu=nmg if i in masked else None,nm_dn=nmd if i in masked else None,rk_gu=rks[i][0],rk_dn=rks[i][1],
           mbn=('exh',i*(ngu+ndn)),var=i%2==1) for i in range(8)]
cov=set()
for e in ex:
    for pj in (e.gu,e.dn):cov|=set((pj.Mb.cpu()*256+pj.Nn.cpu()).tolist())
print(f'(b) (Mb,N) coverage over the 8 experts: {len(cov)}/{nL} valid block words',flush=True)
sel=torch.arange(8,device='cuda')[None];rw=torch.full((1,8),0.125,device='cuda').half();x=(torch.randn(1,H,device='cuda')*0.05).half()
tot=0;bad=0;zs=0
for G in (4,2):
    if G==2:                                             # base variant signs are G = 4 only (T13)
        for e in ex:e.gu.var=e.dn.var=None
    L=MoELayer(8,H,I,nmg,nmd,G=G,mod=Md)
    for lvl in (2,4):
        for e in range(8):L.set(e,ex[e],lvl)
        for e in range(8):
            wg=torch.full((2*I,H),float('nan'),device='cuda').half();wd=torch.full((H,I),float('nan'),device='cuda').half()
            Md.set_wdump(wg.data_ptr(),wd.data_ptr(),e);L(x,sel,rw);torch.cuda.synchronize();Md.set_wdump(0,0,-1)
            for pj,w in ((ex[e].gu,wg),(ex[e].dn,wd)):
                r=dense_W(pj,lvl,G,torch.float16)
                bw=w.view(torch.int16)!=r.view(torch.int16);z=bw&(w==0)&(r==0);zs+=int(z.sum())    # +0 vs -0: same value
                m=bw&~z;eq=not bool(m.any());tot+=1;bad+=not eq
                if not eq:
                    print('  MISMATCH G',G,'lvl',lvl,'e',e,'rk',pj.rk,'mask',pj.fl is not None,'var',pj.var is not None,'n',int(m.sum()),'first',m.nonzero()[:3].tolist())
            del wg,wd
        print(f'  G={G} level {lvl}: done ({tot} projections so far, {bad} mismatching)',flush=True)
    del L
print(f'(b) kernel vs torch ref: {tot} projections, {bad} mismatching (bitwise fp16, +0/-0 treated equal: {zs} signed-zero diffs)')
