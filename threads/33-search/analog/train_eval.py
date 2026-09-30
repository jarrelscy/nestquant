"""train v2-family GBDT (+ analog features) on calib-fit chains 0..23 (every SUB-th block), eval sal-hot/churn on
calib-val (chains 24..31) and glm52-heldout with an hm sweep.
  train_eval.py NAME FEATVAR feat1,feat2,...   (FEATVAR '-' = base only; feats from FEATS9 + an_* + derived)
derived: an_lift = an_f64/max(an_p64,.01)   an_x = an_f64 * mps128   an_d = an_f64 - sema128"""
import os as _o
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _o.environ[_v] = "1"
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

import alib as A

name, fvar, feats = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
SUB = int(os.environ.get("SUB", "4"))
HMS = [float(x) for x in os.environ.get("HMS", "0.3,0.4,0.5,0.7").split(",")]
NLIB = 24
MD = f"{A.OUT}/models"
os.makedirs(MD, exist_ok=True)


def fmat(corpus, L):
    F = A.base_feats(A.load(corpus, L), L, corpus=corpus)
    if fvar != "-":
        z = np.load(f"{A.OUT}/feat/{fvar}/{corpus}/L{L}.npz")
        for k in z.files:
            F[k] = z[k].astype(np.float32) if z[k].ndim == 2 else np.repeat(z[k][:, None], 256, 1).astype(np.float32)
        if "an_f64" in F:
            F["an_lift"] = F["an_f64"] / np.maximum(F["an_p64"], 0.01)
            F["an_x"] = F["an_f64"] * F["mps128"]
            F["an_d"] = F["an_f64"] - F["sema128"]
    return F


def rows_job(L):
    d = A.load("calib-fit", L)
    F = fmat("calib-fit", L)
    Y = A.fut(d["bsal"].astype(np.float64), d["sg"], 64) / A.mL[L]
    nb = Y.shape[0]
    blk = np.arange(nb)
    pick = np.isfinite(Y).all(1) & (blk < NLIB * 512) & (blk % SUB == L % SUB)
    nf = np.ones(256, bool); nf[A.fixed[L]] = False
    X = np.stack([F[n][pick][:, nf] for n in feats], -1).reshape(-1, len(feats))
    return X.astype(np.float32), Y[pick][:, nf].ravel().astype(np.float32)


def eval_job(args):
    L, corpus, model = args
    import lightgbm as lgb
    b = lgb.Booster(model_file=model)
    d = A.load(corpus, L)
    F = fmat(corpus, L)
    S = A.predict(b, F, feats)
    sg = d["sg"]
    if corpus == "calib-fit":
        sg = sg[NLIB:]
    bs = d["bsal"].astype(np.float64)
    out = {}
    for hm in HMS:
        sv, fx = A.sim(S, L, sg, hm=hm)
        m = np.zeros(len(bs), bool)
        for s, e in sg:
            m[s:e] = True
        r = A.metrics(sv[m], fx, bs[m], [(s - sg[0][0], e - sg[0][0]) for s, e in sg])
        out[hm] = r
    return L, corpus, out, (S.astype(np.float16) if corpus == "glm52-heldout" else None)


if __name__ == "__main__":
    import lightgbm as lgb
    t0 = time.time()
    model = os.environ.get("MODEL", f"{MD}/{name}.txt")   # MODEL=path: eval an existing model on FEATVAR feats
    if not os.path.exists(model):
        with Pool(16) as p:
            R = p.map(rows_job, A.LAYERS)
        X = np.concatenate([r[0] for r in R]); y = np.concatenate([r[1] for r in R]); del R
        p = dict(objective="tweedie", metric="tweedie", tweedie_variance_power=1.5, learning_rate=0.3, num_leaves=15,
                 min_data_in_leaf=500, bagging_fraction=0.5, bagging_freq=1, bagging_seed=3, feature_fraction=0.9,
                 seed=0, num_threads=20, verbosity=-1, max_bin=255)
        bst = lgb.train(p, lgb.Dataset(X, y, feature_name=feats, free_raw_data=True),
                        num_boost_round=int(os.environ.get("ITERS", "60")))
        bst.save_model(model)
        imp = dict(zip(feats, bst.feature_importance("gain").round(0).tolist()))
        print(name, "rows", len(y), "train %.0fs" % (time.time() - t0), "gain", imp, flush=True)
        del X, y
    jobs = [(L, c, model) for c in os.environ.get("EVAL", "calib-fit,glm52-heldout").split(",") for L in A.LAYERS]
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(16) as p:          # parent has touched OpenMP (lightgbm): no fork
        R = p.map(eval_job, jobs)
    res = {}
    for L, c, out, S in R:
        res.setdefault(c, {})[L] = out
        if S is not None and os.environ.get("SAVE_SCORES"):
            os.makedirs(f"{A.OUT}/scores/{name}/glm52-heldout", exist_ok=True)
            np.save(f"{A.OUT}/scores/{name}/glm52-heldout/L{L}.npy", S)
    summ = {}
    for c, rl in res.items():
        for hm in HMS:
            s = {k: float(np.mean([rl[L][hm][k] for L in A.LAYERS])) for k in ("sal", "churn", "churn_in")}
            summ[f"{c}@{hm}"] = s
            print(f"{name:16s} {c:14s} hm{hm:.2f} sal-hot {100 * s['sal']:6.2f} churn {s['churn']:5.2f}", flush=True)
    json.dump(dict(feats=feats, fvar=fvar, summ=summ), open(f"{MD}/{name}.eval.json", "w"), indent=1)
