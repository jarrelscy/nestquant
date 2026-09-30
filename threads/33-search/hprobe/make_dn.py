#!/usr/bin/env python3
"""cross-layer router projections (fixed public router weights, no fit): for target layer L and source layer
S = L+k, logits_L(pooled normalised residual of S) = (n_L * hbar_S) @ W_L^T  (linear, so = mean over the block's tokens
of the router-L logits of S's normalised residual).  Features per k: dn{k}_64 (last-64 mean), dn{k}_d (last-64 minus
EMA256).  -> private/feat/dn{k}_64|dn{k}_d/{corpus}/L.npy"""
import os
import sys
from multiprocessing import Pool
import numpy as np
import hplib as HL
import make_feats as M
import t32lib as T

KS = [int(x) for x in sys.argv[1].split(",")]


def job(L):
    R = HL.router()
    W, n = R[f"W{L}"], R[f"n{L}"]
    for k in KS:
        S = min(max(L + k, 3), 77)
        for c in M.CORP:
            h = M.get_h(S, c)
            lg = (h * n) @ W.T
            l64 = HL.bmean(lg, 4)
            M.save(f"dn{k:+d}_64", c, L, l64); M.save(f"dn{k:+d}_d", c, L, l64 - HL.bema(lg, 256))
    return L


if __name__ == "__main__":
    with Pool(8) as p:
        print(len(p.map(job, T.LAYERS)))
