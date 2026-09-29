#!/usr/bin/env python3
"""parity: v2 on CORPUS with heval (target glm52-heldout 74.77 / 2.78)."""
import sys
from multiprocessing import Pool
import numpy as np
import heval as H
import t32lib as T

corpus = sys.argv[1] if len(sys.argv) > 1 else "glm52-heldout"


def job(L):
    d = H.rows(corpus, L)
    S = H.v2_scores(corpus, L, d)
    np.save(f"{H.HP}/private/v2S_{corpus}/L{L}.npy", S)
    return L, H.eval_S(S, L, d["bsal"].astype(np.float64), d["bcnt"].astype(np.float64))


if __name__ == "__main__":
    import os
    os.makedirs(f"{H.HP}/private/v2S_{corpus}", exist_ok=True)
    with Pool(16) as p:
        res = dict(p.map(job, T.LAYERS))
    print(corpus, H.summarise(res))
