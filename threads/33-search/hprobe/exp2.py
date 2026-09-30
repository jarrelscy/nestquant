#!/usr/bin/env python3
"""exp2: v2 GBDT (same params/rows/60 trees) + extra per-(layer, expert) probe features.
Extra features live as [nb, 256] (by expert id) arrays: private/feat/{FEAT}/{corpus}/L{L}.npy (calib-fit ones
out-of-fold by chain).  Training rows = T32 band '' rows (v2's), eval = band all (hot_eval semantics).
  exp2.py train NAME FEAT1,FEAT2 [--split]   (--split: train chains %4 != 3 of calib-fit only -> calib-val model)
  exp2.py eval NAME CORPUS [hm,...] [--val]   (--val: calib-fit chains %4 == 3 only)"""
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
from multiprocessing import Pool  # noqa: E402
import numpy as np  # noqa: E402
import heval as H  # noqa: E402
import hplib as HL  # noqa: E402
import t32lib as T  # noqa: E402

V2F = ["ema32", "ema128", "mem_cur_state", "tok_since_hit", "hits16", "sema32", "sema128", "sal16", "mps128"]
FD = f"{H.HP}/private/feat"
MD = f"{H.HP}/private/models"
K0 = bool(os.environ.get("K0"))
SAVE = bool(os.environ.get("SAVE"))
FDEF77 = {L: list(H.FIXED[L]) + [e for e in H.FDEF[L] if e not in set(H.FIXED[L])] for L in H.FIXED}
PARAMS = dict(objective="tweedie", metric="tweedie", learning_rate=0.3, num_leaves=15, min_data_in_leaf=500,
              bagging_fraction=0.5, bagging_freq=1, bagging_seed=3, feature_fraction=0.9, seed=0, num_threads=18,
              verbosity=-1, max_bin=255, tweedie_variance_power=1.5)


def extra(feats, corpus, L, cand):
    cols = []
    for f in feats:
        A = np.load(f"{FD}/{f}/{corpus}/L{L}.npy").astype(np.float32)
        cols.append(np.take_along_axis(A, cand.astype(np.int64), 1))
    return cols


def _rows(args):
    L, feats, split = args
    d = np.load(f"{T.OUT}/rows/calib-fit/L{L}.npz")
    v = d["valid"].copy()
    if split:
        v &= HL.chain_id(len(v)) % 4 != 3
    X = T.feature_matrix(V2F, "calib-fit", L, band="", d=d).reshape(len(v), -1, len(V2F))
    cols = [X[..., i] for i in range(len(V2F))] + extra(feats, "calib-fit", L, d["cand"])
    Xa = np.stack(cols, -1)[v].reshape(-1, len(cols)).astype(np.float32)
    y = (d["ysal"][v] / HL.mL[str(L)]).ravel().astype(np.float32)
    return Xa, y


def train(name, feats, split):
    import lightgbm as lgb
    t0 = time.time()
    with Pool(8) as p:
        parts = p.map(_rows, [(L, feats, split) for L in T.LAYERS])
    X = np.concatenate([a for a, _ in parts]); y = np.concatenate([b for _, b in parts]); del parts
    print(f"{name}: rows {len(y)} load {time.time() - t0:.0f}s", flush=True)
    names = V2F + feats
    bst = lgb.train(PARAMS, lgb.Dataset(X, y, feature_name=names, free_raw_data=True), num_boost_round=60)
    os.makedirs(MD, exist_ok=True)
    bst.save_model(f"{MD}/{name}.txt")
    imp = dict(zip(names, bst.feature_importance("gain").round(0).tolist()))
    json.dump(dict(feats=feats, split=split, importance_gain=imp), open(f"{MD}/{name}.json", "w"), indent=1)
    print(f"saved {name} {time.time() - t0:.0f}s  gain {imp}", flush=True)


def _eval(args):
    L, name, corpus, hms, val = args
    import lightgbm as lgb
    b = lgb.Booster(model_file=f"{MD}/{name}.txt")
    fn = b.feature_name()
    feats = fn[len(V2F):]
    if K0:                                               # T32 k=0 arm: no fixed set, 77 floating (k0.py)
        d = np.load(f"{T.OUT}/rows_bandk0/{corpus}/L{L}.npz")
        X = np.concatenate([d["X"], np.load(f"{T.OUT}/rows_v2_bandk0/{corpus}/L{L}.npz")["X2"]], -1)
    else:
        d = H.rows(corpus, L)
        X = T.feature_matrix(V2F, corpus, L, band="all", d=d).reshape(d["cand"].shape[0], -1, len(V2F))
    cols = [X[..., i] for i in range(len(V2F))] + extra(feats, corpus, L, d["cand"])
    Xa = np.stack(cols, -1).reshape(-1, len(cols))
    S = T.score_blocks(b.predict(Xa, num_threads=1), d["cand"], d["top"], d["e256"])
    if SAVE:
        os.makedirs(f"{H.HP}/scores/{name}_{corpus}{'_k0' if K0 else ''}", exist_ok=True)
        np.save(f"{H.HP}/scores/{name}_{corpus}{'_k0' if K0 else ''}/L{L}.npy", S.astype(np.float16))
    mask = (HL.chain_id(len(S)) % 4 == 3) if val else None
    bs, bc = d["bsal"].astype(np.float64), d["bcnt"].astype(np.float64)
    if K0:
        out = {}
        for hm in hms:
            sv = T.sim_layer(S, [], FDEF77[L], nf=77, hm=hm, lag=0)
            out[hm] = dict(sn=float((bs * sv).sum()), sd=float(bs.sum()), cn=float((bc * sv).sum()), cd=float(bc.sum()),
                           ch=float((sv[1:] & ~sv[:-1]).sum()), chn=float(len(sv) - 1), ch_in=0.0, chn_in=1.0)
        return L, out
    return L, {hm: H.eval_S(S, L, bs, bc, hm=hm, mask=mask) for hm in hms}


def evaluate(name, corpus, hms, val):
    with Pool(16) as p:
        res = dict(p.map(_eval, [(L, name, corpus, hms, val) for L in T.LAYERS]))
    out = {}
    for hm in hms:
        s = H.summarise({L: res[L][hm] for L in T.LAYERS})
        out[str(hm)] = s
        print(f"{name:24s} {corpus}{'-val' if val else ''}{'-k0' if K0 else ''} hm {hm}: sal {s['sal']:6.2f} churn {s['churn']:5.2f}",
              flush=True)
    f = f"{H.HP}/exp2_results.json"
    allr = json.load(open(f)) if os.path.exists(f) else {}
    allr[f"{name}|{corpus}{'-val' if val else ''}{'-k0' if K0 else ''}"] = out
    json.dump(allr, open(f, "w"), indent=1)


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    if a[0] == "train":
        train(a[1], [f for f in a[2].split(",") if f], "--split" in sys.argv)
    else:
        hms = [float(x) for x in a[3].split(",")] if len(a) > 3 else [0.5]
        evaluate(a[1], a[2], hms, "--val" in sys.argv)
