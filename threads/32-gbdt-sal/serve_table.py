#!/usr/bin/env python3
"""T32 serve-arm table (user request 2026-09-30): all-slot (26 fixed + 51 floating) salience-hot %, routes-hot % and
churn (new floating experts / refresh), hysteresis margin 0.5, sync (lag 0) unless the arm says lag1, mean over L3-77.
  gbdt_x_mps   streaming/gbdt_p64_s5.txt x mps128, native serve band (EMA256 ranks 20-120 scored, top-20 forced)
  gbdt_native  same model without the mps128 rescale
  v2_bandall   streaming/gbdt_v2sal_p64.txt, band all (the T32 reference mode: 74.77 / 2.78 heldout)
  v2_native    same model on the native serve band
  ema          EMA128 hit rate, every non-fixed expert a candidate
  orc_count / orc_sal   top-51 non-fixed by actual next-64-token hits / salience (no hysteresis, uncapped churn)
  serve_table.py CORPUS  -> $OUT/serve_table_CORPUS.json"""
import json
import os
import sys
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402

import t32lib as T  # noqa: E402

corpus = sys.argv[1]
fixed, fdef = T.serve_sets()
S5 = "/home/coder/git/nestquant/streaming/gbdt_p64_s5.txt"
V2 = "/home/coder/git/nestquant/streaming/gbdt_v2sal_p64.txt"
ARMS = [("gbdt_x_mps", S5, "", True, 0), ("gbdt_x_mps_lag1", S5, "", True, 1), ("gbdt_native", S5, "", False, 0),
        ("v2_bandall", V2, "all", False, 0), ("v2_native", V2, "", False, 0), ("ema", None, "all", False, 0)]
NBC = T.CHAIN * T.SEQ // T.G


def oracle(M, fx):
    nb = M.shape[0]
    out = np.zeros(M.shape, bool)
    for c0 in range(0, nb, NBC):
        s = M[c0:c0 + NBC]
        cs = np.vstack([np.zeros((1, 256)), np.cumsum(s, 0)])
        k = np.arange(s.shape[0])
        sc = cs[np.minimum(k + T.H // T.G, s.shape[0])] - cs[k]
        sc[:, fx] = -np.inf
        np.put_along_axis(out[c0:c0 + NBC], np.argsort(-sc, 1, kind="stable")[:, :51], True, 1)
    return out


def job(L):
    import lightgbm as lgb
    rows = {b: np.load(f"{T.OUT}/rows{'_band' + b if b else ''}/{corpus}/L{L}.npz") for b in ("", "all")}
    x2 = {b: np.load(f"{T.OUT}/rows_v2{'_band' + b if b else ''}/{corpus}/L{L}.npz")["X2"] for b in ("", "all")}
    da = rows["all"]
    bc, bs = da["bcnt"].astype(np.float64), da["bsal"].astype(np.float64)
    for b in ("",):
        assert np.array_equal(rows[b]["bcnt"], da["bcnt"])
    fx = np.zeros(256, bool); fx[fixed[L]] = True
    sets = {}
    for name, path, band, mps, lag in ARMS:
        d = rows[band]
        if path is None:
            pred = d["X"][..., list(T.FEATS5).index("ema128")].ravel().astype(np.float64)
        else:
            bst = lgb.Booster(model_file=path)
            nm = bst.feature_name()
            Xm = d["X"].reshape(-1, 5) if tuple(nm) == T.FEATS5 else T.feature_matrix(nm, corpus, L, band=band, d=d)
            pred = bst.predict(Xm, num_threads=1)
        if mps:
            pred = pred * x2[band][..., 3].ravel()
        S = T.score_blocks(pred, d["cand"], d["top"], d["e256"])
        sets[name] = T.sim_layer(S, fixed[L], fdef[L], nf=51, hm=0.5, lag=lag)
    sets["orc_count"] = oracle(bc, fx)
    sets["orc_sal"] = oracle(bs, fx)
    out = {}
    for n, sv in sets.items():
        ch = float((sv[1:] & ~sv[:-1]).sum(1).mean())
        sv = sv.copy(); sv[:, fx] = True
        out[n] = dict(sal=float((bs * sv).sum() / bs.sum()), cnt=float((bc * sv).sum() / bc.sum()), churn=ch)
    return L, out


if __name__ == "__main__":
    with Pool(int(os.environ.get("NPROC", "16"))) as p:
        res = dict(p.map(job, T.LAYERS))
    names = list(res[T.LAYERS[0]])
    summ = {n: {k: float(np.mean([res[L][n][k] for L in T.LAYERS])) for k in ("sal", "cnt", "churn")} for n in names}
    print(f"[{corpus}] all-slot, hm 0.5, mean L3-77")
    for n in names:
        s = summ[n]
        print(f"  {n:16s} sal-hot {s['sal'] * 100:6.2f}  routes-hot {s['cnt'] * 100:6.2f}  churn {s['churn']:5.2f}")
    json.dump(dict(summary=summ, per_layer={str(L): res[L] for L in T.LAYERS}),
              open(f"{T.OUT}/serve_table_{corpus}.json", "w"), indent=1)
