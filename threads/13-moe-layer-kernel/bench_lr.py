"""Low-rank plane cost: full layer us, cold recency routing, B1-4, NQ with lr rank r = 0 / 1 / 4 (r_gu = r_dn = r on
every expert) vs EXL3 1.5.1 coop at the same bit level, plus the pre-lr kernel build (PRE = path to the pre-lr
nqmoe.cu + moe.py pair, default the k2 backups) at r = 0 to show that r = 0 costs nothing.
env I (2048 full | 256 TP8 shard), BS (default 1,2,3,4), LEVELS (comma list of 2, 4p, 4q; default 2,4q), RANKS (default 0,1,4; or rg/rd items),
tune2_I{I}.json per-B cfg2 / cfg4 configs; NQ_STAT=min NQ_BLOCKS=150 recommended on a shared GPU;
PER=1 NQ_TIMING=idle (one graph per routing, each replay timed from idle) keeps every timed replay inside one time slice.  OUT json."""
import torch,os,json,types,sys;torch.cuda.set_per_process_memory_fraction(12/80)
from moe import *;from exl3_moe import *;from common import routings;from timing import bench,warmup
from torch.utils.cpp_extension import load
H,I=6144,int(os.environ.get('I',2048));NP=72 if I>=1024 else 256;R=8
LEV=os.environ.get('LEVELS','2,4q').split(',');RANKS=[r if '/' in r else int(r) for r in os.environ.get('RANKS','0,1,4').split(',')]   # r (r_gu = r_dn = r) or 'rg/rd'
PER=int(os.environ.get('PER',0))
rr=lambda r:tuple(int(v) for v in r.split('/')) if isinstance(r,str) else (r,r)
PRE=os.environ.get('PRE','/tmp/nestquant/13-moe-layer-kernel/k2/')
tune=json.load(open(f'tune2_I{I}.json'))['default']
LV={'2':(2,0,0),'4p':(4,6,7),'4q':(4,0,7)}   # 4q = production level 4 (4.1263 bpw), 4p = 4.0846
pre=None
if PRE and os.path.exists(PRE+'nqmoe.pre_lr.cu'):
    b='/tmp/nestquant/13-moe-layer-kernel/build_prelr';os.makedirs(b,exist_ok=True)
    import shutil;shutil.copy(PRE+'nqmoe.pre_lr.cu',b+'/nqmoe_prelr.cu')
    Mp=load('nqmoe_prelr',[b+'/nqmoe_prelr.cu'],extra_cuda_cflags=['-O3','--use_fast_math','-lineinfo'],build_directory=b,verbose=False)
    src=open(PRE+'moe.pre_lr.py').read().replace('M=get()','M=None')
    pre=types.ModuleType('moe_prelr');pre.__dict__['get']=None;exec(compile(src,'moe_pre_lr.py','exec'),pre.__dict__);pre.M=Mp
exs={k:[Expert(H,I,seed=4000+i,rk_gu=rg,rk_dn=rd,var=True) for i in range(NP)] for k,(lv,rg,rd) in LV.items() if k in LEV}
keep=[];Ls={}
for k in LEV:
    lv=LV[k][0]
    for r in RANKS:
        L=MoELayer(NP,H,I)
        for e,ex in enumerate(exs[k]):
            ex.lr=None;ex.rg=ex.rd=0
            if sum(rr(r)):ex.rand_lr(*rr(r),seed=e);keep.append(ex.lr)
            L.set(e,ex,lv)
        Ls[k,r]=L
    if pre is not None:
        for ex in exs[k]:ex.lr=None;ex.rg=ex.rd=0
        L=pre.MoELayer(NP,H,I,mod=pre.M);[L.set(e,ex,lv) for e,ex in enumerate(exs[k])];Ls[k,'pre']=L
lr_bytes={r:2*(rr(r)[0]*(H+4*I)+rr(r)[1]*(I+2*H)) for r in RANKS}
print('lr bytes/expert',lr_bytes,' expert bytes 4p',exs[LEV[-1]][0].bytes(4),flush=True)
G={2:EXL3Group(NP,2,H=H,I=I,seed=1),4:EXL3Group(NP,4,H=H,I=I,seed=2)}
warmup();out=[]
for B in [int(b) for b in os.environ.get('BS','1,2,3,4').split(',')]:
    rts=routings(B,NP,R);x=(torch.randn(B,H,device='cuda')*0.05).half()
    t=lambda s,lv:tune[f'{s}|B{B}']['cfg2' if lv==2 else 'cfg4']
    fns={};sub=[[(s,w)] for s,w in rts] if PER else [rts]   # PER: one graph per routing (short replays), mean over routings
    for lv in (2,4):
        m=EXL3MoE([(G[lv],0)],B,H=H,I=I,**({} if I>=1024 else dict(smax=256)))
        for j,rs in enumerate(sub):fns[('exl3',lv,j)]=(lambda m=m,rs=rs:[m(x,s,w) for s,w in rs])
    for (k,r),L in Ls.items():
        lv=LV[k][0];cg,cd=t('gu',lv),t('dn',lv)
        for j,rs in enumerate(sub):fns[(k,r,j)]=(lambda L=L,cg=cg,cd=cd,rs=rs:[L(x,s,w,cfg_gu=cg,cfg_dn=cd) for s,w in rs])
    for j in range(len(sub)):fns[('null',j)]=(lambda:x.mul_(1))   # one tiny kernel: graph launch + timing offset
    med,_=bench(fns,blocks=12,repeats=1)
    us=lambda *k:round(sum(med[k+(j,)] for j in range(len(sub)))/R,1)
    row=dict(B=B,null=round(sum(med[('null',j)] for j in range(len(sub)))/R,1),**{f'exl3|{lv}':us('exl3',lv) for lv in (2,4)},**{f'{k}|r{r}':us(k,r) for (k,r) in Ls})
    print(row,flush=True);out.append(row)
json.dump(out,open(os.environ.get('OUT',f'/tmp/nestquant/13-moe-layer-kernel/bench_lr_I{I}.json'),'w'),indent=1)
