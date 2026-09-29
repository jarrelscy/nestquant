#!/usr/bin/env python3
"""hot.py CORPUS [--split val|all] [--hm 0.5,..] NAME=MODEL[@blend...] ... -> sal-hot / churn (T32 hot_eval semantics).
MODEL is a LightGBM file (features by name incl. registered extras)."""
import argparse
import json
import os
import sys
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import dlib as D  # noqa: E402
import extras  # noqa: E402,F401  registers extra feature groups

ap = argparse.ArgumentParser()
ap.add_argument("corpus")
ap.add_argument("models", nargs="+")
ap.add_argument("--split", default="all")
ap.add_argument("--hm", default="0.5")
ap.add_argument("--nproc", type=int, default=16)
ap.add_argument("--save-scores", default="")
ap.add_argument("--out", default="")
a = ap.parse_args()
models = dict(x.split("=", 1) for x in a.models)
HMS = [float(x) for x in a.hm.split(",")]


def job(L):
    import lightgbm as lgb
    import t32lib as T
    d = D.load_rows(a.corpus, L)
    bc, bs = d["bcnt"].astype(np.float64), d["bsal"].astype(np.float64)
    nb = bc.shape[0]
    keep = np.ones(nb, bool) if a.split == "all" else D.chain_split(nb, val=True)
    out = {}
    for name, path in models.items():
        b = lgb.Booster(model_file=path)
        X = D.feats(b.feature_name(), a.corpus, L, d=d)
        pred = b.predict(X.reshape(-1, X.shape[-1]), num_threads=1)
        S = T.score_blocks(pred, d["cand"], d["top"], d["e256"])
        if a.save_scores and name == list(models)[0]:
            os.makedirs(a.save_scores, exist_ok=True)
            np.save(f"{a.save_scores}/L{L}.npy", S.astype(np.float16))
        S, bcx, bsx = S[keep], bc[keep], bs[keep]
        for hm in HMS:
            out[f"{name}@{hm}"] = D.evaluate(S, L, bcx, bsx, hm=hm)
    return L, out


if __name__ == "__main__":
    import t32lib as T
    with Pool(a.nproc) as p:
        res = dict(p.map(job, T.LAYERS))
    summ = {}
    for n in res[T.LAYERS[0]]:
        s = {k: float(np.mean([res[L][n][k] for L in T.LAYERS])) for k in ("sal", "cnt", "churn")}
        summ[n] = s
        print(f"{a.corpus} {a.split} {n:28s} sal-hot {s['sal'] * 100:6.2f}  routes {s['cnt'] * 100:6.2f}  "
              f"churn {s['churn']:5.2f}", flush=True)
    if a.out:
        json.dump({"summary": summ, "per_layer": res}, open(a.out, "w"), indent=1)
