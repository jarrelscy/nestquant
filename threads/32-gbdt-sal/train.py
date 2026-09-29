#!/usr/bin/env python3
"""T32 train: LightGBM on the serve's 5 features, target cnt (routed hits next 64) or sal (sum w^2|x|^2 next 64 / m_L).
  train.py --target sal --obj tweedie:1.5 --out MODEL.txt [--iters 60] [--train calib-fit] [--valid glm52-heldout]
Same family as streaming/gbdt_p64_s5.txt: lr 0.3, 15 leaves, min_data_in_leaf 500, bagging 0.5/1, feature_fraction
0.9; early stopping on the held-out corpus' objective metric, capped at --iters trees (serve latency)."""
import argparse
import json
import os
import time

import numpy as np

import t32lib as T

FEATS = ["ema32", "ema128", "mem_cur_state", "tok_since_hit", "hits16"]


def rows(corpus, target, feats, layers=T.LAYERS, m=None, band="", sub=1):
    Xs, ys, gs = [], [], []
    mL = {}
    for L in layers:
        d = np.load(f"{T.OUT}/rows{'_band' + band if band else ''}/{corpus}/L{L}.npz")
        v = d["valid"]
        X = T.feature_matrix(feats, corpus, L, band=band, valid=True, d=d)
        kind, hz = target[:3], target[3:]                  # e.g. sal, sal32 (rows_x), sal128 / sal256 (rows_lh)
        yx = {"": d, "32": None, "128": None, "256": None}[hz]
        if yx is None:
            yx = np.load(f"{T.OUT}/rows_{'x' if hz == '32' else 'lh'}/{corpus}/L{L}.npz")
        y = yx["y" + kind + hz][v]
        if kind == "sal":
            mL[L] = float(d["slot_sal_sum"] / d["slots"]) if m is None else m[L]
            y = y / mL[L]
        keep = np.isfinite(y).all(1)                         # longer horizons: drop blocks whose horizon leaves chain
        keep &= np.arange(len(keep)) % sub == 0              # --sub: every sub-th valid block (lambdarank cost)
        y = y[keep].ravel()
        X = X.reshape(int(v.sum()), -1, len(feats))[keep].reshape(-1, len(feats))
        v = np.zeros(int(keep.sum()), bool) | True
        Xs.append(X); ys.append(y.astype(np.float32)); gs.append(np.full(int(v.sum()), X.shape[0] // int(v.sum())))
    return np.concatenate(Xs), np.concatenate(ys), mL, np.concatenate(gs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True,
                    choices=["cnt", "sal", "cnt32", "sal32", "cnt128", "sal128", "cnt256", "sal256"])
    ap.add_argument("--obj", default="poisson")
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--lr", type=float, default=0.3)
    ap.add_argument("--leaves", type=int, default=15)
    ap.add_argument("--train", default="calib-fit")
    ap.add_argument("--valid", default="glm52-heldout")
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--out", required=True)
    ap.add_argument("--v2", action="store_true", help="+ salience features t32lib.FEATS_V2 (serve change)")
    ap.add_argument("--v3", action="store_true", help="+ router-prob features t32lib.FEATS_V3 (build_v3.py)")
    ap.add_argument("--band", default="", help="rows band ('' = EMA256 ranks 20-120, all, all20; build.py T32_BAND)")
    ap.add_argument("--sub", type=int, default=1, help="train on every sub-th valid block only")
    ap.add_argument("--nbins", type=int, default=8, help="lambdarank: graded salience bins (0 = y==0, rest by "
                    "train quantiles of y>0; label_gain = bin mean y / top bin mean * 31)")
    ap.add_argument("--feats", default="", help="explicit comma list of feature names (overrides --v2/--v3)")
    a = ap.parse_args()
    import lightgbm as lgb
    t0 = time.time()
    feats = FEATS + (list(T.FEATS_V2) if a.v2 else []) + (list(T.FEATS_V3) if a.v3 else [])
    if a.feats:
        feats = a.feats.split(",")
    Xt, yt, m, gt = rows(a.train, a.target, feats, band=a.band, sub=a.sub)
    Xv, yv, _, gv = rows(a.valid, a.target, feats, m=m if a.target.startswith("sal") else None, band=a.band)
    print(f"rows train {len(yt)} valid {len(yv)}  load {time.time() - t0:.0f}s  y mean {yt.mean():.4f} "
          f"zero frac {(yt == 0).mean():.3f}", flush=True)
    obj, _, pw = a.obj.partition(":")
    p = dict(objective=obj, metric="ndcg" if obj == "lambdarank" else obj, learning_rate=a.lr, num_leaves=a.leaves, min_data_in_leaf=500,
             bagging_fraction=0.5, bagging_freq=1, bagging_seed=3, feature_fraction=0.9, seed=0,
             num_threads=a.threads, verbosity=-1, max_bin=255)
    if obj == "tweedie":
        p["tweedie_variance_power"] = float(pw or 1.5)
    kw_t, kw_v, rank = {}, {}, None
    if obj == "lambdarank":             # query = (layer, block): the candidates of one refresh; graded salience labels
        q = np.quantile(yt[yt > 0], np.linspace(0, 1, a.nbins)[1:-1])
        lab = lambda y: np.where(y > 0, 1 + np.searchsorted(q, y, side="right"), 0).astype(np.int32)  # noqa: E731
        lt = lab(yt)
        gain = [float(yt[lt == k].mean()) if (lt == k).any() else 0.0 for k in range(a.nbins)]
        gain = [31.0 * g / gain[-1] for g in gain]
        p.update(label_gain=gain, lambdarank_truncation_level=64, eval_at=[51], min_data_in_leaf=500)
        rank = dict(bins=q.tolist(), label_gain=gain)
        yt, yv = lt, lab(yv)
        kw_t, kw_v = dict(group=gt), dict(group=gv)
    dt = lgb.Dataset(Xt, yt, feature_name=feats, free_raw_data=True, **kw_t)
    dv = lgb.Dataset(Xv, yv, reference=dt, **kw_v)
    ev = {}
    bst = lgb.train(p, dt, num_boost_round=a.iters, valid_sets=[dv], valid_names=["heldout"],
                    callbacks=[lgb.early_stopping(10, verbose=False), lgb.record_evaluation(ev),
                               lgb.log_evaluation(10)])
    bi = bst.best_iteration or a.iters
    bst.save_model(a.out, num_iteration=bi)
    meta = dict(features=feats, target=a.target, obj=a.obj, params=p, best_iteration=bi, iters_cap=a.iters,
                valid_curve=ev["heldout"][list(ev["heldout"])[0]], train=a.train, valid=a.valid, band=a.band,
                n_train=int(len(yt)), sub=a.sub, n_valid=int(len(yv)), sal_norm_mL=m if a.target.startswith("sal") else None, rank=rank,
                wall_s=time.time() - t0)
    json.dump(meta, open(a.out + ".meta.json", "w"), indent=1)
    print(f"saved {a.out} best_iter {bi} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
