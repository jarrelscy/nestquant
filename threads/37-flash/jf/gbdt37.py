#!/usr/bin/env python3
"""T37 v2 GBDT retrain (port of threads/32-gbdt-sal/train.py --v2 --target sal --obj tweedie:1.5 --iters 60).
Same family and params as /tmp/nestquant/32-gbdt-sal/models/v2_sal_tweedie1.5.txt: 9 serve features (5 count +
4 salience, jlib37.V2F), target = salience of the next 64 tokens / m_L, Tweedie 1.5, lr 0.3, 15 leaves,
min_data_in_leaf 500, bagging 0.5/1, feature_fraction 0.9, <= 60 trees, early stopping (10) on the val split.
Differences: rows = every J37_GSUB-th block x ALL non-fixed experts of the Flash train split (GLM-5.3: EMA256 ranks
20..120 of calib-fit), because jF applies v2 to all experts; m_L stored as sal_norm_mL in the meta, as before.
Mixed runs: train rows = decode + the prefill minority (feat37 mix); early stopping on DECODE val rows only.
  gbdt37.py [--threads 20] [--out $OUT/models/v2_sal_tweedie1.5.txt]"""
import argparse
import json
import os
import time

import numpy as np

import jlib37 as J


def rows(sp, kind=None):
    """kind None: all rows; 0: decode rows only (early stopping = decode-only validation)."""
    Xs, ys, ks, m = [], [], [], {}
    for L in J.LAYERS:
        z = np.load(f"{J.OUT}/gbdt_rows/{sp}/L{L}.npz")
        kd = z["kind"] if "kind" in z.files else np.zeros(len(z["y"]), np.int8)
        sel = slice(None) if kind is None else kd == kind
        Xs.append(z["X"][sel]); ys.append(z["y"][sel]); ks.append(kd[sel]); m[L] = float(z["mL"])
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(ks), m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--lr", type=float, default=0.3)
    ap.add_argument("--leaves", type=int, default=15)
    ap.add_argument("--min-data", type=int, default=500)
    ap.add_argument("--threads", type=int, default=20)
    ap.add_argument("--out", default=J.V2)
    a = ap.parse_args()
    import lightgbm as lgb
    t0 = time.time()
    Xt, yt, kt, m = rows("train")
    Xv, yv, _, _ = rows("val", kind=0)
    pf = float((kt == 1).mean()) if len(kt) else 0.0
    print(f"[gbdt37] rows train {len(yt)} (prefill {100 * pf:.1f}%) val (decode only) {len(yv)} load "
          f"{time.time() - t0:.0f}s y mean {yt.mean():.4f} zero frac {(yt == 0).mean():.3f}", flush=True)
    p = dict(objective="tweedie", metric="tweedie", learning_rate=a.lr, num_leaves=a.leaves, min_data_in_leaf=a.min_data,
             bagging_fraction=0.5, bagging_freq=1, bagging_seed=3, feature_fraction=0.9, seed=0,
             num_threads=a.threads, verbosity=-1, max_bin=255, tweedie_variance_power=1.5)
    dt = lgb.Dataset(Xt, yt, feature_name=list(J.V2F), free_raw_data=True)
    dv = lgb.Dataset(Xv, yv, reference=dt)
    ev = {}
    bst = lgb.train(p, dt, num_boost_round=a.iters, valid_sets=[dv], valid_names=["val"],
                    callbacks=[lgb.early_stopping(10, verbose=False), lgb.record_evaluation(ev), lgb.log_evaluation(10)])
    bi = bst.best_iteration or a.iters
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    bst.save_model(a.out, num_iteration=bi)
    meta = dict(features=list(J.V2F), target="sal", obj="tweedie:1.5", params=p, best_iteration=bi, iters_cap=a.iters,
                valid_curve=ev["val"]["tweedie"], train="t37 train split (decode + prefill mix)",
                valid="t37 val split, decode rows only", train_prefill_frac=pf,
                band="all non-fixed", gsub=int(os.environ.get("J37_GSUB", "16")), n_train=int(len(yt)), n_valid=int(len(yv)),
                sal_norm_mL={str(k): v for k, v in m.items()}, model="GLM-5.3-Flash", NE=J.NE, layers=J.LAYERS,
                wall_s=time.time() - t0)
    json.dump(meta, open(a.out + ".meta.json", "w"), indent=1)
    print(f"[gbdt37] saved {a.out} best_iter {bi} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
