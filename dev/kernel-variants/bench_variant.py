"""A/B a kernel build variant against the default build on one real layer (one TP4 rank): outputs must match the default
build (bitwise, or to fp32 atomic-order noise if the default build itself is not bitwise repeatable), then both are
timed interleaved in the same bench call. Routing model as bench_real.py; level 4 = fixed set + next most routed up to N4.
  bench_variant.py ROOT L DEFS(comma list, e.g. NQ_PREFETCH) [N4s=26,128] [Bs=1,4,8] [rank=0] [out.json]
BV_SAME=1: every call uses the same routing (L2-resident weights): if time barely drops, the kernel is not DRAM-bound."""
import os,sys,json,torch,numpy as np
HERE=os.path.dirname(os.path.abspath(__file__));sys.path[:0]=[HERE,HERE+'/../streaming']
torch.cuda.set_per_process_memory_fraction(20/96)
import build,nqload as NQ,fixed_set as FS
from moe import MoELayer
from timing import bench,warmup
root,L,defs=sys.argv[1],int(sys.argv[2]),[d for d in sys.argv[3].split(',') if d];a=sys.argv[4:]+[None]*4
N4s=[int(x) for x in (a[0] or '26,128').split(',')];Bs=[int(x) for x in (a[1] or '1,4,8').split(',')];rank=int(a[2] or 0);out=a[3]
CFG_GU=[None,[1,8,4],[1,8,6],[1,8,6],[1,8,6],[1,8,6],[1,8,6],[1,8,12],[1,8,12]]
CFG_DN=[None,[1,8,4],[1,8,2],[1,8,4],[1,8,4],[1,8,4],[1,8,4],[1,8,4],[1,8,4]]
m0=build.get([]);m1=build.get(defs)
NE=256;R=8
RL=NQ.RankLayer(root,L,rank,4);H,I=RL.H,RL.I
fr=np.array(json.load(open(HERE+'/../threads/22-boundary-experts/fixed_set.json'))['n_routed'][str(L)],float);p=fr/fr.sum()
fx,_,_=FS.load(layers=[L])
def rows(B,rng):
    rs=[]
    for t in range(B):
        r=[e for e in rs[-1] if rng.random()<0.4] if t else []
        while len(r)<8:
            e=int(rng.choice(NE,p=p))
            if e not in r:r.append(e)
        rs.append(r)
    return rs
warmup();res=[];ok=True
for N4 in N4s:
    l4=list(fx[L])+[int(e) for e in np.argsort(-fr) if e not in fx[L]][:max(0,N4-len(fx[L]))]
    Ms=[MoELayer(NE,H,I,Bmax=8,mod=m) for m in (m0,m1)]
    for E in range(NE):
        lv=4 if E in l4 else 2;ex=RL.ex[E];ex.signs=ex.sc[lv]
        for M in Ms:M.set(E,ex,lv)
        ex.signs=ex.sc[2]
    for B in Bs:
        rng=np.random.default_rng(B);rts=[]
        for i in range(R):
            s=torch.tensor(rows(B,rng),device='cuda');w=torch.rand(B,8,device='cuda')+0.1;rts.append((s,(w/w.sum(1,keepdim=True)).half()))
        if os.environ.get('BV_SAME')=='1':rts=[rts[0]]*R      # L2-warm probe: the same routing every call (weights stay in L2)
        x=(torch.randn(B,H,device='cuda')*0.05).half();cg,cd=CFG_GU[B],CFG_DN[B]
        eq=True;rep=True;err=0.
        for s,w in rts:
            y0=Ms[0](x,s,w,cfg_gu=cg,cfg_dn=cd).clone();y0b=Ms[0](x,s,w,cfg_gu=cg,cfg_dn=cd).clone();y1=Ms[1](x,s,w,cfg_gu=cg,cfg_dn=cd).clone()
            rep&=torch.equal(y0,y0b);eq&=torch.equal(y0,y1);err=max(err,((y1-y0).norm()/y0.norm()).item())
        good=eq or (not rep and err<1e-5);ok&=good
        fns={'base':(lambda M=Ms[0]:[M(x,s,w,cfg_gu=cg,cfg_dn=cd) for s,w in rts]),'var':(lambda M=Ms[1]:[M(x,s,w,cfg_gu=cg,cfg_dn=cd) for s,w in rts])}
        med,_=bench(fns,blocks=16,repeats=1)
        r=dict(L=L,B=B,n4=N4,share4=round(float(p[l4].sum()),3),defs=defs,base_us=round(med['base']/R,2),var_us=round(med['var']/R,2),
               gain_pct=round(100*(1-med['var']/med['base']),2),bitwise=eq,base_repeatable=rep,max_rel=err,match=good)
        res.append(r);print(json.dumps(r),flush=True)
    del Ms;torch.cuda.empty_cache()
print('VARIANT MATCH' if ok else 'VARIANT MISMATCH')
if out:json.dump(res,open(out,'w'),indent=1)
