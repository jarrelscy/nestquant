import torch,sys;torch.cuda.set_per_process_memory_fraction(12/80)
from nq15 import *
for vid in range(int(sys.argv[1]) if len(sys.argv)>1 else 0,53):
    p=Proj(vid,256,512)
    for cfg in [(1,8,1),(2,4,2),(1,4,3)]:
        if p.K%(cfg[0]*cfg[2]*128):continue
        r=check(p,3,cfg)
        print(vid,M.info(vid)[:2],cfg,'yrel %.2e Wmax %.2e Wrel %.2e std %.3f'%r,'bpw %.3f'%p.bpw,flush=True)
