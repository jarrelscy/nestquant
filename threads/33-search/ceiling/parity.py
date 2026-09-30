#!/usr/bin/env python3
"""T33l parity: v2 hm0.5 (74.77/2.78) and oracle next-64 cap3 (85.6) on CORPUS."""
import json, os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
from multiprocessing import Pool
import numpy as np
import clib as C
corpus = sys.argv[1]


def job(L):
    d = np.load(f"{C.T.OUT}/rows_bandall/{corpus}/L{L}.npz")
    bs = d["bsal"].astype(np.float64)
    fx, fd = C.masks(L)
    S = C.v2_scores(corpus, L)
    O4 = C.fut(bs, 4)
    arms = {"v2_hm0.5": C.replay(S, fx, fd, hm=0.5), "v2_hm0.4": C.replay(S, fx, fd, hm=0.4),
            "orc64": C.replay(O4, fx, fd), "orc64_cap3": C.replay(O4, fx, fd, cap=3),
            "orc64_hm_cap3": C.replay(O4, fx, fd, hm=0.5, cap=3)}
    return L, {n: C.metric(sv, fx, bs) for n, sv in arms.items()}


if __name__ == "__main__":
    with Pool(int(os.environ.get("NPROC", "20"))) as p:
        res = dict(p.map(job, C.T.LAYERS))
    out = {}
    for n in res[3]:
        s = float(np.mean([res[L][n][0] for L in C.T.LAYERS])); c = float(np.mean([res[L][n][1] for L in C.T.LAYERS]))
        out[n] = (s, c)
        print(f"{corpus} {n:16s} sal-hot {s*100:6.2f} churn {c:5.2f}", flush=True)
    json.dump(out, open(f"{C.OUTC}/parity_{corpus}.json", "w"), indent=1)
