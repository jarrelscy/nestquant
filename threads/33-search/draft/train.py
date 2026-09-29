#!/usr/bin/env python3
"""train.py --feats a,b,.. --out M.txt : LightGBM Tweedie-1.5 salience target (T32 v2 recipe: lr 0.3, 15 leaves,
min_data 500, bagging 0.5, ff 0.9, <=60 trees), band-all rows; train = calib-fit chains %4 != 3, early stopping on
calib-fit val chains (%4 == 3).  heldout never touched."""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import dlib as D  # noqa: E402
import extras  # noqa: E402,F401
import t32lib as T  # noqa: E402
from multiprocessing import Pool  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--feats", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--iters", type=int, default=60)
ap.add_argument("--leaves", type=int, default=15)
ap.add_argument("--lr", type=float, default=0.3)
ap.add_argument("--threads", type=int, default=16)
ap.add_argument("--full", action="store_true", help="train on all calib-fit (no val; fixed iters from --iters)")
a = ap.parse_args()
feats = a.feats.split(",")


def job(L):
    d = D.load_rows("calib-fit", L)
    X = D.feats(feats, "calib-fit", L, d=d)
    v = d["valid"]
    y = d["ysal"] / float(d["slot_sal_sum"] / d["slots"])
    tr = D.chain_split(len(v)) & v
    va = D.chain_split(len(v), val=True) & v
    if a.full:
        tr = v.copy()
    return (X[tr].reshape(-1, len(feats)), y[tr].ravel().astype(np.float32),
            X[va].reshape(-1, len(feats)), y[va].ravel().astype(np.float32))


if __name__ == "__main__":
    import lightgbm as lgb
    t0 = time.time()
    with Pool(8) as p:
        R = p.map(job, T.LAYERS)
    Xt = np.concatenate([r[0] for r in R]); yt = np.concatenate([r[1] for r in R])
    Xv = np.concatenate([r[2] for r in R]); yv = np.concatenate([r[3] for r in R])
    del R
    print(f"rows train {len(yt)} val {len(yv)} load {time.time() - t0:.0f}s", flush=True)
    p = dict(objective="tweedie", tweedie_variance_power=1.5, metric="tweedie", learning_rate=a.lr,
             num_leaves=a.leaves, min_data_in_leaf=500, bagging_fraction=0.5, bagging_freq=1, bagging_seed=3,
             feature_fraction=0.9, seed=0, num_threads=a.threads, verbosity=-1, max_bin=255)
    dt = lgb.Dataset(Xt, yt, feature_name=feats, free_raw_data=True)
    dv = lgb.Dataset(Xv, yv, reference=dt)
    ev = {}
    cb = [lgb.record_evaluation(ev), lgb.log_evaluation(10)] + ([] if a.full else [lgb.early_stopping(10, verbose=False)])
    bst = lgb.train(p, dt, num_boost_round=a.iters, valid_sets=[dv], valid_names=["val"], callbacks=cb)
    bi = bst.best_iteration or a.iters
    bst.save_model(a.out, num_iteration=bi)
    json.dump(dict(features=feats, params=p, best_iteration=bi, curve=ev["val"]["tweedie"], wall=time.time() - t0),
              open(a.out + ".meta.json", "w"), indent=1)
    print(f"saved {a.out} best_iter {bi} {time.time() - t0:.0f}s", flush=True)
