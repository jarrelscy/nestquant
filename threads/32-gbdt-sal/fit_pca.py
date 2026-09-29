#!/usr/bin/env python3
"""T32 M1: per-layer PCA basis from the capture's aggregate hidden covariance (nq_e2e NQ_TRACE_HID, cov mode).
  fit_pca.py COVDIR OUTDIR K   -> OUTDIR/pca_L{li}.npz (mean [H], V [H,K], explained var ratio)  (PRIVATE dir)"""
import glob
import os
import re
import sys

import numpy as np

src, dst, k = sys.argv[1], sys.argv[2], int(sys.argv[3])
os.makedirs(dst, exist_ok=True)
for L in sorted({int(re.search(r"hidcov_L(\d+)\.", f).group(1)) for f in glob.glob(f"{src}/hidcov_L*.npz")}):
    n = 0; s = ss = None
    for f in sorted(glob.glob(f"{src}/hidcov_L{L}.r*.npz")):
        z = np.load(f)
        n += int(z["n"]); s = z["s"] if s is None else s + z["s"]; ss = z["ss"] if ss is None else ss + z["ss"]
    mu = s / n
    cov = ss / n - np.outer(mu, mu)
    ev, V = np.linalg.eigh(cov)
    o = np.argsort(ev)[::-1]
    ev, V = ev[o], V[:, o]
    np.savez(f"{dst}/pca_L{L}.npz", mean=mu.astype(np.float32), V=V[:, :k].astype(np.float32),
             evr=(ev[:k] / ev.sum()).astype(np.float32), n=n)
    print(f"L{L} n {n} H {len(mu)} top-{k} explained {ev[:k].sum() / ev.sum():.3f} (top1 {ev[0] / ev.sum():.3f})")
