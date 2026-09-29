#!/usr/bin/env python3
"""T32 co-activation stats from the TRAIN split only (calib-fit trace), per token:
  C[L]  [256,256]  P(e routed at L | j routed at L-1)       (row j normalised; L = 4..77; sparse layers only)
  A[L]  [256,256]  P(i routed at L | j routed at L), diag 0  (column j normalised: A[i, j])
-> /tmp/nestquant/32-gbdt-sal/stats/coact.npz  (aggregate 256x256 counts only; still kept private)"""
import os
import numpy as np
import t32lib as T

ids = {L: T.load_layer(L, "calib-fit")[0].astype(np.int64) for L in T.LAYERS}
C, A = {}, {}
for L in T.LAYERS:
    x = ids[L]
    pair = (x[:, :, None] * 256 + x[:, None, :]).ravel()
    n = np.bincount(pair, minlength=65536).reshape(256, 256).astype(np.float64)
    np.fill_diagonal(n, 0)
    hj = np.bincount(x.ravel(), minlength=256).astype(np.float64)
    A[L] = (n / np.maximum(hj, 1)[None, :]).astype(np.float32)     # A[i,j] = n(i&j)/n(j)
    if L - 1 in ids:
        p = ids[L - 1]
        pair = (p[:, :, None] * 256 + x[:, None, :]).ravel()
        m = np.bincount(pair, minlength=65536).reshape(256, 256).astype(np.float64)
        hp = np.bincount(p.ravel(), minlength=256).astype(np.float64)
        C[L] = (m / np.maximum(hp, 1)[:, None]).astype(np.float32)  # C[j,e] = n(j@L-1 & e@L)/n(j@L-1)
os.makedirs(f"{T.OUT}/stats", exist_ok=True)
np.savez(f"{T.OUT}/stats/coact.npz", **{f"C{L}": v for L, v in C.items()}, **{f"A{L}": v for L, v in A.items()})
print("C rows sum (should be 8)", float(C[40].sum(1)[C[40].sum(1) > 0].mean()), "A col sum (7)", float(A[40].sum(0).mean()))
