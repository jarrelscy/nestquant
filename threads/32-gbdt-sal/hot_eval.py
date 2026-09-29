#!/usr/bin/env python3
"""T32 arm report metric: all-slot salience-hot % (26 fixed + 51 floating) and routes-hot %, sync (lag 0),
hysteresis 1.5, band all, churn = new floating experts per refresh; mean over layers (+ bands).  Same numbers as
decomp.py gbdt_sync_hm (v2 heldout 74.8 / churn 2.78).
  hot_eval.py CORPUS NAME=MODEL.txt ...   -> $OUT/hot_eval_CORPUS.json (merged by name)"""
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
BANDS = {"L3-6": range(3, 7), "L7-40": range(7, 41), "L41-77": range(41, 78)}


def job(L):
    import lightgbm as lgb
    d = np.load(f"{T.OUT}/rows_bandall/{corpus}/L{L}.npz")
    bc, bs = d["bcnt"].astype(np.float64), d["bsal"].astype(np.float64)
    out = {}
    for name, path in models.items():
        b = lgb.Booster(model_file=path)
        S = T.score_blocks(b.predict(T.feature_matrix(b.feature_name(), corpus, L, band="all", d=d), num_threads=1),
                           d["cand"], d["top"], d["e256"])
        sv = T.sim_layer(S, fixed[L], fdef[L], nf=51, hm=0.5, lag=0)
        ch = float((sv[1:] & ~sv[:-1]).sum(1).mean())
        sv[:, fixed[L]] = True
        out[name] = dict(sal=float((bs * sv).sum() / bs.sum()), cnt=float((bc * sv).sum() / bc.sum()), churn=ch)
    return L, out


if __name__ == "__main__":
    with Pool(int(os.environ.get("NPROC", "16"))) as p:
        res = dict(p.map(job, T.LAYERS))
    f = f"{T.OUT}/hot_eval_{corpus}.json"
    allr = json.load(open(f)) if os.path.exists(f) else {}
    for n in models:
        s = {k: float(np.mean([res[L][n][k] for L in T.LAYERS])) for k in ("sal", "cnt", "churn")}
        for bn, r in BANDS.items():
            s[bn] = float(np.mean([res[L][n]["sal"] for L in r]))
        s["per_layer_sal"] = [res[L][n]["sal"] for L in T.LAYERS]
        s["model"] = models[n]
        allr[n] = s
        print(f"{n:14s} sal-hot {s['sal'] * 100:6.2f}  routes {s['cnt'] * 100:6.2f}  churn {s['churn']:5.2f}  " +
              "  ".join(f"{bn} {s[bn] * 100:5.1f}" for bn in BANDS), flush=True)
    json.dump(allr, open(f, "w"), indent=1)
