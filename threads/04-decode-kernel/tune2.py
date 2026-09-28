import torch,json,sys,os;torch.cuda.set_per_process_memory_fraction(12/80)
from nq2 import *
from timing import bench
NC=6
CANDS=[('T2H',2),('B2',1),('B2',2),('B2',4),('T4',1),('T4',2),('A3',2),('A3',4),('A4',2),('A4',4),('MIX',2),('MIX',4)]
out=json.load(open('tune2.json')) if os.path.exists('tune2.json') else {}
for d,G in CANDS:
    for (N,K,nm) in [(4096,6144,'gu'),(6144,2048,'down')]:
        ps=[Proj2(d,N,K,G=G) for _ in range(NC)]
        for B in [1,4]:
            k=f"{d}_G{G}|{nm}|{B}"
            if k in out:continue
            x=torch.randn(B,K,device='cuda').half();y=torch.zeros(B,N,device='cuda')
            cf=ps[0].configs()
            med,_=bench({c:(lambda c=c:[p(x,y,c) for p in ps]) for c in cf},blocks=15,repeats=1)
            c=min(med,key=med.get);out[k]=dict(cfg=list(c),us=med[c]/NC,gbs=ps[0].bytes/(med[c]/NC)/1e3)
            print(k,c,round(med[c]/NC,2),'us',round(out[k]['gbs']),'GB/s',flush=True)
            json.dump(out,open('tune2.json','w'),indent=1)
        del ps;torch.cuda.empty_cache()
