"""train a LightGBM floating-set model on g33rows rows.
  g33train.py --rows band --feats base9|all|LIST --out NAME [--obj tweedie:1.5] [--iters 60] [--lr 0.3] [--leaves 15]
     [--mdl 500] [--ff 0.9] [--dart] [--layers 3-77] [--weight none|bnd:A:W] [--init v2] [--sub 1] [--es 20]
weight bnd:A:W  row weight 1 + A * exp(-|rank_v2 - 51| / W)  (salience at stake near the rank-51 boundary)
init v2         boost from the v2 model's log-prediction (stacking: residual trees on top of v2)"""
import argparse
import json
import os
import time
import numpy as np
import glib as g

ap = argparse.ArgumentParser()
ap.add_argument("--rows", default="band"); ap.add_argument("--feats", default="base9"); ap.add_argument("--out", required=True)
ap.add_argument("--obj", default="tweedie:1.5"); ap.add_argument("--iters", type=int, default=60)
ap.add_argument("--lr", type=float, default=0.3); ap.add_argument("--leaves", type=int, default=15)
ap.add_argument("--mdl", type=int, default=500); ap.add_argument("--ff", type=float, default=0.9)
ap.add_argument("--bag", type=float, default=0.5)
ap.add_argument("--dart", action="store_true"); ap.add_argument("--layers", default="3-77")
ap.add_argument("--weight", default="none"); ap.add_argument("--init", default="")
ap.add_argument("--sub", type=int, default=1); ap.add_argument("--es", type=int, default=20)
ap.add_argument("--l2", type=float, default=0.0); ap.add_argument("--maxbin", type=int, default=255)
ap.add_argument("--threads", type=int, default=20)
a = ap.parse_args()
import lightgbm as lgb  # noqa: E402

t0 = time.time()
COLS = g.ALLX + ["v2"]
feats = {"base9": g.BASE9, "all": g.ALLX}.get(a.feats, None) or a.feats.split(",")
ci = [COLS.index(f) for f in feats]
lo, hi = map(int, a.layers.split("-"))
R = f"{g.W}/rows/{a.rows}"


def load(k):
    X = np.load(f"{R}/{k}_X.npy", mmap_mode="r"); y = np.load(f"{R}/{k}_y.npy"); aux = np.load(f"{R}/{k}_aux.npy")
    m = (aux[:, 0] >= lo) & (aux[:, 0] <= hi)
    if a.sub > 1 and k == "tr":
        m &= aux[:, 2] % a.sub == 0
    idx = np.flatnonzero(m)
    Xs = np.empty((len(idx), len(ci)), np.float32)
    for j, c in enumerate(ci):
        Xs[:, j] = X[idx, c]
    v2 = np.asarray(X[idx, COLS.index("v2")]) if a.init == "v2" else None
    w = None
    if a.weight.startswith("bnd"):
        _, A, Wd = a.weight.split(":")
        w = 1 + float(A) * np.exp(-np.abs(aux[idx, 3].astype(np.float32) - 51) / float(Wd))
    return Xs, y[idx], w, v2


Xt, yt, wt, v2t = load("tr"); Xv, yv, wv, v2v = load("va")
print(f"rows tr {len(yt)} va {len(yv)} feats {len(feats)} load {time.time() - t0:.0f}s", flush=True)
obj, _, pw = a.obj.partition(":")
p = dict(objective=obj, metric=obj, learning_rate=a.lr, num_leaves=a.leaves, min_data_in_leaf=a.mdl,
         bagging_fraction=a.bag, bagging_freq=1, bagging_seed=3, feature_fraction=a.ff, seed=0, lambda_l2=a.l2,
         num_threads=a.threads, verbosity=-1, max_bin=a.maxbin)
if obj == "tweedie":
    p["tweedie_variance_power"] = float(pw or 1.5)
if a.dart:
    p.update(boosting="dart", drop_rate=0.1, skip_drop=0.5)
it = lambda v: None if v is None else np.log(np.maximum(v, 1e-12))  # noqa: E731
dt = lgb.Dataset(Xt, yt, weight=wt, init_score=it(v2t), feature_name=feats, free_raw_data=True)
dv = lgb.Dataset(Xv, yv, weight=wv, init_score=it(v2v), reference=dt)
ev = {}
cb = [lgb.record_evaluation(ev), lgb.log_evaluation(25)]
if not a.dart:
    cb.append(lgb.early_stopping(a.es, verbose=False))
bst = lgb.train(p, dt, num_boost_round=a.iters, valid_sets=[dv], valid_names=["val"], callbacks=cb)
bi = (bst.best_iteration or a.iters) if not a.dart else a.iters
os.makedirs(f"{g.W}/models", exist_ok=True)
path = f"{g.W}/models/{a.out}.txt"
bst.save_model(path, num_iteration=bi)
curve = ev["val"][list(ev["val"])[0]]
meta = dict(vars(a), features=feats, params=p, best_iteration=bi, val_best=float(min(curve)), val_curve=curve[::5],
            n_train=int(len(yt)), wall_s=time.time() - t0)
json.dump(meta, open(path + ".meta.json", "w"), indent=1)
print(f"saved {path} best_iter {bi} val {min(curve):.5f} {time.time() - t0:.0f}s", flush=True)
