import torch,random
from routing import *
def routings(B,NP,R,mode='recency',seed=0):
    """R routings, each avoiding the previous call's experts (cold L2 for weights between consecutive calls)."""
    rng=random.Random(seed*100+B);out=[];prev=()
    for i in range(R):
        rows=make_sel(B,range(NP),mode,rng=rng,avoid=prev);prev=[e for r in rows for e in r];out.append(routing_tensors(rows,seed=i))
    return out
