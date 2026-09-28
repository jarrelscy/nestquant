"""A/B in one process (interleaved graphs, cold L2): pre-port decoder (nqk2 A4, /tmp backup build) vs nqdec (thread 15)
at fixed cfgs, per stage. env I, VARS as tune2.py (new-decoder build variants)."""
import torch,sys,os,types,importlib.util,json;torch.cuda.set_per_process_memory_fraction(12/80)
import build as B_;from torch.utils.cpp_extension import load;from timing import bench,warmup;from common import routings
T='/tmp/nestquant/13-moe-layer-kernel/oldchk'
Mold=load('nqmoeold',[T+'/nqmoe.cu'],extra_cuda_cflags=['-O3','--use_fast_math','-lineinfo'],build_directory='/tmp/nestquant/13-moe-layer-kernel/buildold',verbose=False)
fake=types.ModuleType('build');fake.get=lambda defs=None:Mold;sys.modules['build']=fake
sp=importlib.util.spec_from_file_location('moe_old',T+'/moe.py');mo=importlib.util.module_from_spec(sp);sp.loader.exec_module(mo)
sys.modules['build']=B_
import moe as mn
H,I=6144,int(os.environ.get('I',2048));NP=72 if I>=1024 else 256;R=8
VARS=os.environ.get('VARS','').split(';')
mods={v:B_.get([d for d in v.split(',') if d]) for v in VARS}
exo=[mo.Expert(H,I,seed=i) for i in range(NP)];exn=[mn.Expert(H,I,seed=i) for i in range(NP)]
Ls={}
for lv in (2,4):
    L=mo.MoELayer(NP,H,I,G=2);[L.set(e,exo[e],lv) for e in range(NP)];Ls['old',lv]=L
    for v in VARS:
        for G in (2,4):
            L=mn.MoELayer(NP,H,I,G=G,mod=mods[v]);[L.set(e,exn[e],lv) for e in range(NP)];Ls[(v,G),lv]=L
CF=json.loads(os.environ.get('CFGS','[[1,4,8],[1,8,8],[2,8,4],[2,8,8],[1,8,6]]'))
warmup()
for Bt in [1,2,3,4]:
    rts=routings(Bt,NP,R);x=(torch.randn(Bt,H,device='cuda')*0.05).half()
    for part,which in (('gu',1),('dn',2)):
        K=H if part=='gu' else I;cs=[c for c in CF if K%(c[0]*c[2]*128)==0]
        fns={(k,lv,tuple(c)):(lambda L=L,c=c:[L(x,s,w,which=which,cfg_gu=list(c),cfg_dn=list(c)) for s,w in rts]) for (k,lv),L in Ls.items() for c in cs}
        med,_=bench(fns,blocks=10,repeats=1)
        for lv in (2,4):
            row={str(k):round(min(med[(k,lv,tuple(c))] for c in cs)/R,1) for k in dict.fromkeys(kk for kk,_ in Ls)}
            print(part,'B',Bt,'L',lv,row,flush=True)
