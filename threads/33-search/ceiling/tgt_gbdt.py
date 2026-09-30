#!/usr/bin/env python3
"""T33l: does a lower-noise training target help?  v2 recipe (9 v2 features, tweedie 1.5, lr 0.3, 15 leaves, 60 trees)
trained on the 64 calib-fit prefixes (x 75 layers x 256 experts at the decision block gk) with target
  E   = mean over the K=8 sampled continuations of next-64 salience / m_L
  R   = realised (real text) next-64 salience / m_L
evaluated on the 64 heldout prefixes, k=0 (77 floating), dm0 (blk16) and hold64 (<=12.8 swaps from v2's k0 set).
  tgt_gbdt.py GEN_DIR"""
import os
import sys
os.environ["KF"] = "0"
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, "/home/coder/git/nestquant/threads/33-search/scale")
from multiprocessing import Pool
import numpy as np
import ceil_eval as E
import scalelib as SL
C, T, A = E.C, E.T, E.A


def fjob(L):
    """features F[gk] [nq, 256, 9] per corpus in `order` positions + v2 parity + incumbents (E.job)."""
    import lightgbm as lgb
    out = {}
    for corpus in ("calib-fit", "glm52-heldout"):
        qs = [q for q, i in enumerate(E.order) if E.PX[i]["corpus"] == corpus]
        D = SL.load(corpus, L)
        F, _ = SL.feats(D)
        gks = [E.PX[E.order[q]]["gk"] for q in qs]
        out[corpus] = (np.array(qs), F[gks].astype(np.float32))
    b = lgb.Booster(model_file=C.V2)
    S = np.load(f"{A.CACHE}/glm52-heldout/L{L}.npz")["S"]
    qs, Fh = out["glm52-heldout"]
    p = b.predict(Fh.reshape(-1, 9), num_threads=1).reshape(len(qs), 256)
    ref = S[[E.PX[E.order[q]]["gk"] for q in qs]]
    par = float(np.abs(p - ref).max() / max(np.abs(ref).max(), 1e-12))
    mL = float(np.load(f"{A.CACHE}/calib-fit/L{L}.npz")["bsal"].astype(np.float64).sum()) / (
        np.load(f"{A.CACHE}/calib-fit/L{L}.npz")["bsal"].shape[0] * T.G * 8)
    _, inc = E.job(L)
    return L, (out, par, mL, inc)


def main():
    Y16, Y64 = E.ymats(16), E.ymats(64)
    with Pool(int(os.environ.get("NPROC", "4"))) as p:
        R = dict(p.map(fjob, E.SL))
    print("v2 parity max rel |pred - cached S| =", max(R[L][1] for L in E.SL), flush=True)
    K, K1 = E.K, E.K1
    real = np.arange(E.NP) * K1 + K
    Xtr, yE, yR, Xte = [], [], [], []
    for L in E.SL:
        j = E.SL.index(L)
        out, _, mL, _ = R[L]
        qc, Fc = out["calib-fit"]
        smp = qc[:, None] * K1 + np.arange(K)[None]
        Xtr.append(Fc.reshape(-1, 9))
        yE.append((Y64[j][smp].mean(1) / mL).ravel())
        yR.append((Y64[j][real[qc]] / mL).ravel())
    Xtr = np.concatenate(Xtr); yE = np.concatenate(yE).astype(np.float32); yR = np.concatenate(yR).astype(np.float32)
    print(f"train rows {len(yE)}  yE mean {yE.mean():.4f} zero {(yE == 0).mean():.3f} | yR mean {yR.mean():.4f} "
          f"zero {(yR == 0).mean():.3f}", flush=True)
    import lightgbm as lgb
    feats = ["ema32", "ema128", "mem_cur_state", "tok_since_hit", "hits16", "sema32", "sema128", "sal16", "mps128"]
    models = {}
    for tag, y in (("E", yE), ("R", yR)):
        for seed in (0, 1, 2):
            prm = dict(objective="tweedie", tweedie_variance_power=1.5, learning_rate=0.3, num_leaves=15,
                       min_data_in_leaf=500, bagging_fraction=0.5, bagging_freq=1, bagging_seed=3 + seed,
                       feature_fraction=0.9, seed=seed, num_threads=int(os.environ.get("NPROC", "4")), verbosity=-1,
                       max_bin=255)
            models[f"{tag}{seed}"] = lgb.train(prm, lgb.Dataset(Xtr, y, feature_name=feats), num_boost_round=60)
    # ---- evaluate on heldout prefixes, k0
    arms = ["v2"] + [f"{t}{s}" for t in "ER" for s in range(3)]
    acc = {a: [] for a in arms}
    per = {a: [] for a in arms}          # per (layer) dm0 blk16 per-prefix hit shares for paired SE
    for L in E.SL:
        j = E.SL.index(L)
        out, _, mL, inc_d = R[L]
        qh, Fh = out["glm52-heldout"]
        fx, _ = E.masks(L)
        inc = np.stack([inc_d[q][0] for q in qh])
        S2 = np.stack([inc_d[q][2] for q in qh])
        y16, y64 = Y16[j, real[qh]], Y64[j, real[qh]]
        for a in arms:
            V = S2 if a == "v2" else models[a].predict(Fh.reshape(-1, 9), num_threads=4).reshape(len(qh), 256)
            row = []
            for dm in E.DMS:
                m = E.pick(V, inc, fx, dm)
                row.append((E.share(m, fx, y16), E.share(m, fx, y64), float((m & ~inc).sum(1).mean())))
            acc[a].append(row)
            m0 = E.pick(V, inc, fx, 0.0)
            per[a].append(((y16 * (m0 | fx[None])).sum(1) / np.maximum(y16.sum(1), 1e-30)))
    res = {}
    for a in arms:
        g = np.array(acc[a]).mean(0)
        h64, f = C.at_churn([(x[1], x[2]) for x in g], 12.8)
        res[a] = (100 * g[0, 0], 100 * g[0, 1], 100 * h64)
        print(f"k0 heldout N={len(qh)}  {a:3s} dm0 blk16 {100*g[0,0]:6.2f} next64 {100*g[0,1]:6.2f} | hold64 {100*h64:6.2f} {f}",
              flush=True)
    for t in "ER":
        v = np.mean([res[f"{t}{s}"] for s in range(3)], 0)
        print(f"mean-of-3-seeds {t}: dm0 blk16 {v[0]:6.2f} next64 {v[1]:6.2f} hold64 {v[2]:6.2f}", flush=True)
    dE = np.mean([np.array(per[f"E{s}"]) for s in range(3)], 0).mean(0)      # per-prefix (mean over layers)
    dR = np.mean([np.array(per[f"R{s}"]) for s in range(3)], 0).mean(0)
    d = dE - dR
    print(f"paired E-R dm0 blk16 per-prefix: {100*d.mean():+.2f} +- {100*d.std(ddof=1)/np.sqrt(len(d)):.2f} (SE, N={len(d)})",
          flush=True)


if __name__ == "__main__":
    main()
