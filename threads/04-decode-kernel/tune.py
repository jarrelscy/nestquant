import torch,json,sys;torch.cuda.set_per_process_memory_fraction(12/80)
from nq import *
from timing import bench
NC=6
CANDS=[('J4',1,0),('J4',0,0),('J3',1,0)]
_OLD=[('UNI2',1,0),('T2',1,0),('H2',1,0),('H2',1,1),('H2B',1,0),('T2R1',1,0),
       ('UNI4',1,0),('T4',1,0),('H4',1,0),('H4',1,1),('T2R2',1,0),('T2R2',0,0),('T2H',1,0),('T2H',0,0),('T2H',1,1),('T2H',0,1)]
def key(d,sp,lr):return f"{d}{'' if sp else '_inter'}{'_lutr' if lr else ''}"
best={}
for d,sp,lr in CANDS:
    for (N,K,nm) in [(4096,6144,'gu'),(6144,2048,'down')]:
        ps=[Proj(d,N,K,split=bool(sp),lutr=bool(lr)) for _ in range(NC)]
        for B in [1,4]:
            x=torch.randn(B,K,device='cuda').half();y=torch.zeros(B,N,device='cuda')
            cf=ps[0].configs()
            fns={c:(lambda c=c:[p(x,y,c) for p in ps]) for c in cf}
            med,_=bench(fns,blocks=15,repeats=1)
            c=min(med,key=med.get);best[f"{key(d,sp,lr)}|{nm}|{B}"]=dict(cfg=c,us=med[c]/NC,gbs=ps[0].bytes/(med[c]/NC)/1e3)
            print(key(d,sp,lr),nm,B,c,round(med[c]/NC,2),'us',round(ps[0].bytes/(med[c]/NC)/1e3),'GB/s',flush=True)
        del ps;torch.cuda.empty_cache()
best={**json.load(open('tune.json')),**best}
json.dump({k:dict(v,cfg=list(v['cfg'])) for k,v in best.items()},open('tune.json','w'),indent=1)
