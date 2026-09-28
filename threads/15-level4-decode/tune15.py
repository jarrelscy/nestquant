"""Tune (cpw, sb, nst) for fused-B chain parts: gate/up mode 3 [4096 x 6144], down mode 4 [6144 x 2048]."""
import torch,json,os,sys;torch.cuda.set_per_process_memory_fraction(12/80)
from nq15 import *
from timing import bench
NC=6
sg=lambda n:(torch.randint(0,2,(n,),device='cuda')*2-1).half()
sv_g,sv_u,su_d,su_x,sv_d=sg(2048),sg(2048),sg(2048),sg(6144),sg(6144)
F='tune15.json';out=json.load(open(F)) if os.path.exists(F) else {}
vids=[int(v) for v in sys.argv[1].split(',')]
for vid in vids:
    gu=[Proj(vid,4096,6144,seed=i) for i in range(NC)];dn=[Proj(vid,6144,2048,seed=10+i) for i in range(NC)]
    for B in [1,4]:
        if f'{vid}|gu|{B}' in out:continue
        x32=torch.randn(B,6144,device='cuda')*0.01;acc=[torch.zeros(B,4096,device='cuda') for _ in range(2)]
        hh=torch.zeros(B,2048,device='cuda').half();y=torch.zeros(B,6144,device='cuda');yo=torch.empty(B,6144,device='cuda').half()
        c2=torch.zeros(48,dtype=torch.int32,device='cuda')
        fg={('gu',c):(lambda c=c:[p(x32,acc[0],c,mode=3,ex=(su_x,)) for p in gu]) for c in gu[0].configs()}
        fd={('down',c):(lambda c=c:[p(hh,y,c,mode=4,ex=(acc[0],acc[1],sv_g,sv_u,su_d,sv_d,yo,c2)) for p in dn]) for c in dn[0].configs()}
        med,_=bench({**fg,**fd},blocks=20,repeats=1)
        for part in ['gu','down']:
            ks=[k for k in med if k[0]==part];k=min(ks,key=med.get)
            out[f'{vid}|{part}|{B}']=dict(cfg=list(k[1]),us=med[k]/NC);print(vid,part,B,k[1],round(med[k]/NC,2),flush=True)
        json.dump(out,open(F,'w'),indent=1)
    del gu,dn;torch.cuda.empty_cache()
