import torch;torch.cuda.set_per_process_memory_fraction(12/80)
from nq import *
for d in DEC:
    for split in ([True,False] if d in PLANES else [True]):
        for lutr in ([False,True] if d in LUTD else [False]):
            p=Proj(d,512,1024,split=split,lutr=lutr,r=4 if d.startswith('SYN') else 0)
            for cfg in [(1,4,2,1),(2,2,2,2),(1,8,1,4),(1,8,1,3)]:
                if cfg not in p.configs():continue
                print(d,'split' if split else 'inter','lutr' if lutr else '',cfg,['%.2e'%v for v in check(p)] if False else check(p) if True else 0)
