"""Per-stage launch-config sweep (cold L2, recency routing), gu (which=1) and dn (which=2) separately."""
import torch,json,os,itertools;torch.cuda.set_per_process_memory_fraction(12/80)
from moe import *;from common import *;from timing import bench,warmup
H,I=6144,int(os.environ.get('I',2048));NP=72;R=8
ex=[Expert(H,I,seed=i) for i in range(NP)]
Ls={}
for lv in [2,4]:
    L=MoELayer(NP,H,I);[L.set(e,ex[e],lv) for e in range(NP)];Ls[lv]=L
fn=f'tune_I{I}.json';out={}
def cfgs(K,N):
    r=[]
    for cpw,sb,nst in itertools.product([1,2],[4,8],[2,3,4,6,8]):
        if K%(cpw*nst*128) or (N//16)%sb or 128%(sb*16):continue
        r.append((cpw,sb,nst))
    return r
warmup()
for B in [1,2,3,4]:
    rts=routings(B,NP,R);x=(torch.randn(B,H,device='cuda')*0.05).half()
    for part,which,K,N in [('gu',1,H,2*I),('dn',2,I,H)]:
        fns={(lv,c):(lambda c=c,L=Ls[lv]:[L(x,s,w,which=which,cfg_gu=list(c),cfg_dn=list(c)) for s,w in rts]) for lv in [2,4] for c in cfgs(K,N)}
        med,_=bench(fns,blocks=8,repeats=1)
        cs=cfgs(K,N);tot={c:med[(2,c)]+med[(4,c)] for c in cs}
        best=min(tot,key=tot.get)
        out[f'{part}|B{B}']=dict(cfg=list(best),us2=med[(2,best)]/R,us4=med[(4,best)]/R,best2=min(med[(2,c)] for c in cs)/R,best4=min(med[(4,c)] for c in cs)/R)
        print(part,B,out[f'{part}|B{B}'],flush=True)
        json.dump(out,open(fn,'w'),indent=1)
