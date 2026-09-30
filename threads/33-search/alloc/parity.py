#!/usr/bin/env python3
"""parity: v2 sync hm0.5 k=26 all-slot sal-hot / routes / churn from the alloc cache (target heldout 74.77 / 2.78)."""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from multiprocessing import Pool
import alib as A

corpus = sys.argv[1] if len(sys.argv) > 1 else "glm52-heldout"


def job(L):
    S, bs, bc = A.load(corpus, L)
    sv = A.sim(S, A.fixed[L], A.fdef[L], 51, 0.5)
    ch = A.churn(sv)
    sv[:, A.fixed[L]] = True
    return float((bs * sv).sum() / bs.sum()), float((bc * sv).sum() / bc.sum()), ch


if __name__ == "__main__":
    with Pool(16) as p:
        r = np.array(p.map(job, A.T.LAYERS))
    print(f"{corpus}: sal-hot {r[:,0].mean()*100:.2f} routes {r[:,1].mean()*100:.2f} churn {r[:,2].mean():.2f}")
