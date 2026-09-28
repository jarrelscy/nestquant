"""Routing generators. 'distinct': no expert shared between tokens (worst case, 8B runs).
'recency': token t reuses each of token t-1's experts w.p. p (GLM MTP window: ~21 distinct at B4, thread 10)."""
import random,torch
def make_sel(B,pool,mode='recency',p=0.4,rng=None,avoid=()):
    rng=rng or random.Random(0);pool=[e for e in pool if e not in set(avoid)]
    rows=[]
    used=set()
    for t in range(B):
        row=[]
        if t and mode=='recency':
            row=[e for e in rows[-1] if rng.random()<p]
        cand=[e for e in pool if e not in row and (mode!='distinct' or e not in used)]
        rng.shuffle(cand);row+=cand[:8-len(row)];rng.shuffle(row)
        rows.append(row);used|=set(row)
    return rows
def routing_tensors(rows,device='cuda',seed=0):
    g=torch.Generator().manual_seed(seed)
    sel=torch.tensor(rows,dtype=torch.long)
    w=torch.rand(sel.shape,generator=g)+0.1;w=w/w.sum(1,keepdim=True)
    return sel.to(device),w.half().to(device)
