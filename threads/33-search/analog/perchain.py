"""per-chain sal-hot (heldout / sm120tf) for models on feature variants.  perchain.py CORPUS HM name=model@featvar ..."""
import os as _o
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _o.environ[_v] = "1"
import json
import sys
import multiprocessing as mp
import numpy as np
import alib as A

corpus, HMS = sys.argv[1], [float(x) for x in sys.argv[2].split(",")]
arms = {a.split("=")[0]: a.split("=")[1].split("@") for a in sys.argv[3:]}


def fm(fvar, L):
    F = A.base_feats(A.load(corpus, L), L, corpus=corpus)
    if fvar != "-":
        z = np.load(f"{A.OUT}/feat/{fvar}/{corpus}/L{L}.npz")
        for k in z.files:
            F[k] = z[k].astype(np.float32) if z[k].ndim == 2 else np.repeat(z[k][:, None], 256, 1).astype(np.float32)
    return F


def job(L):
    import lightgbm as lgb
    d = A.load(corpus, L); bs = d["bsal"].astype(np.float64)
    out = {}
    for n, (m, fv) in arms.items():
        b = lgb.Booster(model_file=m)
        S = A.predict(b, fm(fv, L), b.feature_name())
        for HM in HMS:
            sv, fx = A.sim(S, L, d["sg"], hm=HM)
            hot = sv | fx
            ch = (sv[1:] & ~sv[:-1]).sum(1)
            out[(n, HM)] = [[float((bs[s:e] * hot[s:e]).sum()), float(bs[s:e].sum()), float(ch[s:e - 1].sum()), e - s - 1]
                            for s, e in d["sg"]]
    return L, out


if __name__ == "__main__":
    with mp.get_context("spawn").Pool(16) as p:
        R = dict(p.map(job, A.LAYERS))
    for n, HM in [(n, h) for h in HMS for n in arms]:
        a = np.array([R[L][(n, HM)] for L in A.LAYERS])          # [L, chain, 4]
        per = (a[..., 0] / a[..., 1]).mean(0) * 100
        print(f"{A.LAYOUT} hm{HM} {n:14s} all {per.mean() if False else (a[..., 0] / a[..., 1]).mean(0).mean() * 100:6.2f} "
              f"pooled-per-layer {((a[..., 0].sum(1) / a[..., 1].sum(1)).mean() * 100):6.2f} churn "
              f"{(a[..., 2].sum(1) / a[..., 3].sum(1)).mean():.2f} | ex-last {((a[:, :-1, 0].sum(1) / a[:, :-1, 1].sum(1)).mean() * 100):6.2f} "
              f"churn {(a[:, :-1, 2].sum(1) / a[:, :-1, 3].sum(1)).mean():.2f} | per chain " + " ".join(f"{x:5.1f}" for x in per))
