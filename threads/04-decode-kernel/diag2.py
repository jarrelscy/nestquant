import torch;torch.cuda.set_per_process_memory_fraction(12/80)
from timing import bench
NC=6
res={}
for mb in [3.1,6.3,12.6,25.2,100]:
    n=int(mb*1e6/4)
    src=[torch.randn(n,device='cuda') for _ in range(NC)];dst=[torch.empty(n,device='cuda') for _ in range(NC)]
    o=torch.empty((),device='cuda')
    fns={f'sum{mb}':lambda:[torch.sum(s,dim=0,out=o) for s in src],f'copy{mb}':lambda:[d.copy_(s) for d,s in zip(dst,src)]}
    med,r=bench(fns,blocks=30,repeats=1)
    for k,v in med.items():
        us=v/NC;res[k]=us;print(k,round(us,2),'us',round((mb*(2 if 'copy' in k else 1))*1e6/us/1e3),'GB/s')
    del src,dst
