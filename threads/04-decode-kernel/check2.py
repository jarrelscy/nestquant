import torch;torch.cuda.set_per_process_memory_fraction(12/80)
from nq import *
for d in ['J4','J3','T4']:
    for split in [True,False] if d!='T4' else [True]:
        p=Proj(d,512,1024,split=split);print(d,split,check(p),check(p) if False else '')
