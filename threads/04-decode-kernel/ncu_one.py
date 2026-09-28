import torch,sys;torch.cuda.set_per_process_memory_fraction(12/80)
from nq import *
d=sys.argv[1];cfg=tuple(int(v) for v in sys.argv[2].split(','));N,K=int(sys.argv[3]),int(sys.argv[4]);B=int(sys.argv[5]) if len(sys.argv)>5 else 1
ps=[Proj(d,N,K) for _ in range(6)];x=torch.randn(B,K,device='cuda').half();y=torch.zeros(B,N,device='cuda')
for i in range(12):ps[i%6](x,y,cfg)
torch.cuda.synchronize()
