"""Per-stage launch-config sweep across kernel build variants, interleaved in one process (cold L2, recency routing).
env: I (2048 | 256), VARS = ';'-separated build variants, each ','-separated -D macros ('' = default build).
Writes tune2_I{I}.json: per variant, per part (gu = which 1, dn = which 2), per B: the cfg minimising us2+us4, plus
per-level bests."""
import torch,json,os,itertools;torch.cuda.set_per_process_memory_fraction(12/80)
from moe import *;from common import *;from timing import bench,warmup
from build import get
H,I=6144,int(os.environ.get('I',2048));NP=72 if I>=1024 else 256;R=8
VARS=os.environ.get('VARS','').split(';')
mods={v:get([d for d in v.split(',') if d]) for v in VARS}
ex=[Expert(H,I,seed=i) for i in range(NP)]
Ls={}
for v in VARS:
    for lv in [2,4]:
        L=MoELayer(NP,H,I,mod=mods[v]);[L.set(e,ex[e],lv) for e in range(NP)];Ls[v,lv]=L
fn=f'tune2_I{I}.json';out={}
def cfgs(K,N):
    r=[]
    for cpw,sb,nst in itertools.product([1,2],[2,4,8],[1,2,3,4,6,8]):
        if K%(cpw*nst*128) or (N//16)%sb or 128%(sb*16):continue
        r.append((cpw,sb,nst))
    return r
warmup()
for B in [1,2,3,4]:
    rts=routings(B,NP,R);x=(torch.randn(B,H,device='cuda')*0.05).half()
    for part,which,K,N in [('gu',1,H,2*I),('dn',2,I,H)]:
        cs=cfgs(K,N)
        fns={(v,lv,c):(lambda c=c,L=Ls[v,lv]:[L(x,s,w,which=which,cfg_gu=list(c),cfg_dn=list(c)) for s,w in rts]) for v in VARS for lv in [2,4] for c in cs}
        med,_=bench(fns,blocks=8,repeats=1)
        for v in VARS:
            tot={c:med[(v,2,c)]+med[(v,4,c)] for c in cs};best=min(tot,key=tot.get)
            b2=min(cs,key=lambda c:med[(v,2,c)]);b4=min(cs,key=lambda c:med[(v,4,c)])
            out.setdefault(v or 'default',{})[f'{part}|B{B}']=dict(cfg=list(best),us2=med[(v,2,best)]/R,us4=med[(v,4,best)]/R,
                cfg2=list(b2),best2=med[(v,2,b2)]/R,cfg4=list(b4),best4=med[(v,4,b4)]/R)
            print(v or 'default',part,B,{k:(round(a,1) if isinstance(a,float) else a) for k,a in out[v or 'default'][f'{part}|B{B}'].items()},flush=True)
        json.dump(out,open(fn,'w'),indent=1)
