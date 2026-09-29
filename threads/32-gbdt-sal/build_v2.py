#!/usr/bin/env python3
"""T32 v2 rows: salience features (t32lib.v2_features) next to rows/CORPUS/L.npz -> rows_v2/CORPUS/L.npz"""
import os
import sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
from multiprocessing import Pool  # noqa: E402
import numpy as np  # noqa: E402
import t32lib as T  # noqa: E402
corpus = sys.argv[1]
band = os.environ.get("T32_BAND", "")
sfx = "_band" + band if band else ""
out = f"{T.OUT}/rows_v2{sfx}/{corpus}"
os.makedirs(out, exist_ok=True)


def job(L):
    d = np.load(f"{T.OUT}/rows{sfx}/{corpus}/L{L}.npz")
    np.savez(f"{out}/L{L}.npz", X2=T.v2_features(d["bcnt"], d["bsal"], d["cand"]))
    return L


if __name__ == "__main__":
    with Pool(40) as p:
        print(sorted(p.map(job, T.LAYERS))[-1])
