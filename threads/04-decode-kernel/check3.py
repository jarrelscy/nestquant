import torch;torch.cuda.set_per_process_memory_fraction(12/80)
from nq2 import *
for d in ['B2','T4','A3','A4','MIX']:
    for G in [1,2,4]:
        p=Proj2(d,64,512,G=G)
        for cfg in [(1,4,2),(2,2,1)]:print(d,G,cfg,[round(v,5) for v in check(p,cfg=cfg)])
