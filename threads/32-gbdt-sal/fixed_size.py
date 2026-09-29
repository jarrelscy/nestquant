#!/usr/bin/env python3
"""T32 idea 8: fixed-set size.  77 level-4 slots per layer = k fixed + (77 - k) floating, k in {0, 13, 26}; sync mode
(lag 0), band-all rows (every expert a candidate, so rows do not depend on the fixed set).  The k fixed = top-k of the
manifest's 26 fixed by calib-fit salience (the manifest stores them unranked; REAP is a calib statistic too); floating
start set = remaining fixed-26 by salience, then floating_default order.  Metric: all-slot hot % (fixed + floating) of
routed salience and routes, mean over layers + bands.
  fixed_size.py CORPUS NAME=MODEL ...  -> $OUT/fixed_size_CORPUS.json"""
import json
import os
import sys
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402

import t32lib as T  # noqa: E402

corpus = sys.argv[1]
models = dict(x.split("=", 1) for x in sys.argv[2:])
fixed, fdef = T.serve_sets()
KS = (0, 13, 26)
BANDS = {"L3-6": range(3, 7), "L7-40": range(7, 41), "L41-77": range(41, 78)}


def job(L):
    import lightgbm as lgb
    d = np.load(f"{T.OUT}/rows_bandall/{corpus}/L{L}.npz")
    cal = np.load(f"{T.OUT}/rows_bandall/calib-fit/L{L}.npz")["bsal"].sum(0)
    f26 = sorted(fixed[L], key=lambda e: -cal[e])
    bc, bs = d["bcnt"].astype(np.float64), d["bsal"].astype(np.float64)
    out = {}
    for name, path in models.items():
        b = lgb.Booster(model_file=path)
        S = T.score_blocks(b.predict(T.feature_matrix(b.feature_name(), corpus, L, band="all", d=d), num_threads=1),
                           d["cand"], d["top"], d["e256"])
        for k in KS:
            fx = f26[:k]
            sv = T.sim_layer(S, fx, f26[k:] + list(fdef[L]), nf=77 - k, hm=0.5, lag=0)
            sv[:, fx] = True
            out[f"{name}_k{k}"] = dict(sal=float((bs * sv).sum() / bs.sum()), cnt=float((bc * sv).sum() / bc.sum()),
                                       churn=float((sv[1:] & ~sv[:-1]).sum(1).mean()))
    return L, out


if __name__ == "__main__":
    with Pool(int(os.environ.get("NPROC", "16"))) as p:
        res = dict(p.map(job, T.LAYERS))
    summ = {}
    for n in res[T.LAYERS[0]]:
        summ[n] = {k: float(np.mean([res[L][n][k] for L in T.LAYERS])) for k in ("sal", "cnt", "churn")}
        for bn, r in BANDS.items():
            summ[n][bn] = [float(np.mean([res[L][n][k] for L in r])) for k in ("sal", "cnt")]
        print(f"{n:14s} hot sal {summ[n]['sal']:.4f} routes {summ[n]['cnt']:.4f} churn {summ[n]['churn']:.2f}  " +
              "  ".join(f"{bn} {v[0]:.3f}/{v[1]:.3f}" for bn, v in summ[n].items() if bn.startswith("L")))
    json.dump({"corpus": corpus, "models": models, "summary": summ, "per_layer": res},
              open(f"{T.OUT}/fixed_size_{corpus}.json", "w"), indent=1)
