"""Tune configs for the fused-B 2-launch chain: gate/up in mode 3 (input WHT prologue), down in mode 4 (SwiGLU prologue + output WHT epilogue)."""
import torch,json,os;torch.cuda.set_per_process_memory_fraction(12/80)
from nq2 import *
from timing import bench
NC=6
sg=lambda n:(torch.randint(0,2,(n,),device='cuda')*2-1).half()
sv_g,sv_u,su_d,su_x,sv_d=sg(2048),sg(2048),sg(2048),sg(6144),sg(6144)
out=json.load(open('tune_fused.json')) if os.path.exists('tune_fused.json') else {}
for d,G,kw in [('T2H',2,{}),('B2',2,{}),('A4',2,{}),('A4',4,{}),('MIX',2,dict(grp=8)),('A3',2,{}),('T4',2,{})]:
    nm=f"{d}{'g8' if kw else ''}_G{G}"
    gu=[Proj2(d,4096,6144,G=G,**kw) for _ in range(NC)];dn=[Proj2(d,6144,2048,G=G,**kw) for _ in range(NC)]
    for B in [1,4]:
        if f'{nm}|gu|{B}' in out:continue
        x32=torch.randn(B,6144,device='cuda')*0.01;acc=[torch.zeros(B,4096,device='cuda') for _ in range(2)]
        hh=torch.zeros(B,2048,device='cuda').half();y=torch.zeros(B,6144,device='cuda');yo=torch.empty(B,6144,device='cuda').half()
        c2=torch.zeros(48,dtype=torch.int32,device='cuda')
        fg={('gu',c):(lambda c=c:[p(x32,acc[0],c,mode=3,ex=(su_x,)) for p in gu]) for c in gu[0].configs() if c[1]==8 or c[1]==4}
        fd={('down',c):(lambda c=c:[p(hh,y,c,mode=4,ex=(acc[0],acc[1],sv_g,sv_u,su_d,sv_d,yo,c2)) for p in dn]) for c in dn[0].configs()}
        med,_=bench({**fg,**fd},blocks=20,repeats=1)
        for part in ['gu','down']:
            ks=[k for k in med if k[0]==part];k=min(ks,key=med.get)
            out[f'{nm}|{part}|{B}']=dict(cfg=list(k[1]),us=med[k]/NC);print(nm,part,B,k[1],round(med[k]/NC,2),flush=True)
        json.dump(out,open('tune_fused.json','w'),indent=1)
    del gu,dn;torch.cuda.empty_cache()
