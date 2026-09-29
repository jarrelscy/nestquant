"""MoE layer on real weights split by launch: K1 (gate|up + SwiGLU finisher) alone, K2 (down + combine) alone, both;
grid-mapped vs persistent (cfg[3] = blocks per SM multiplier). Same routing model as bench_real.py.
  bench_split.py ROOT L [rank=0] [N4s=26,128] [B=4] [out.json]"""
import os,sys,json,itertools,torch,numpy as np
HERE=os.path.dirname(os.path.abspath(__file__));sys.path[:0]=[HERE,HERE+'/../streaming']
torch.cuda.set_per_process_memory_fraction(20/96)
import nqload as NQ,fixed_set as FS
from moe import MoELayer
from timing import bench,warmup
root,L=sys.argv[1],int(sys.argv[2]);a=sys.argv[3:]+[None]*4
rank=int(a[0] or 0);N4s=[int(x) for x in (a[1] or '26,128').split(',')];B=int(a[2] or 4);out=a[3]
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
warmup();res=[]
rng=np.random.default_rng(B);rts=[]
for i in range(R):
    s=torch.tensor(rows(B,rng),device='cuda');w=torch.rand(B,8,device='cuda')+0.1;rts.append((s,(w/w.sum(1,keepdim=True)).half()))
x=(torch.randn(B,H,device='cuda')*0.05).half()
ndist=np.mean([len(set(s.flatten().tolist())) for s,_ in rts])
for N4 in N4s:
    l4=list(fx[L])+[int(e) for e in np.argsort(-fr) if e not in fx[L]][:max(0,N4-len(fx[L]))]
    M=MoELayer(NE,H,I,Bmax=8)
    for E in range(NE):
        lv=4 if E in l4 else 2;ex=RL.ex[E];ex.signs=ex.sc[lv];M.set(E,ex,lv);ex.signs=ex.sc[2]
    share4=float(p[l4].sum())
    cg0,cd0=[1,8,6],[1,8,4]
    cfgs={'both':(3,cg0,cd0),'k1':(1,cg0,cd0),'k2':(2,cg0,cd0)}
    for m in (1,2,4):
        cfgs[f'both_p{m}']=(3,cg0+[m],cd0+[m]);cfgs[f'k1_p{m}']=(1,cg0+[m],cd0);cfgs[f'k2_p{m}']=(2,cg0,cd0+[m])
    for cg in ([1,8,3],[1,8,12],[2,8,3],[1,4,6]):cfgs[f'k1_{cg}']=(1,cg,cd0)
    for cd in ([1,8,2],[1,8,1],[1,4,4],[2,8,2]):cfgs[f'k2_{cd}']=(2,cg0,cd)
    fns={k:(lambda wh=wh,cg=cg,cd=cd:[M(x,s,w,which=wh,cfg_gu=cg,cfg_dn=cd) for s,w in rts]) for k,(wh,cg,cd) in cfgs.items()}
    med,_=bench(fns,blocks=12,repeats=1)
    r=dict(L=L,rank=rank,B=B,n4=N4,share4=round(share4,3),distinct=float(ndist),us={k:round(v/R,1) for k,v in med.items()})
    res.append(r);print(json.dumps(r),flush=True)
    del M;torch.cuda.empty_cache()
if out:json.dump(res,open(out,'w'),indent=1)
