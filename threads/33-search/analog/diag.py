"""quick signal diagnostic: next-64 non-fixed salience captured by top-51 (no hysteresis) of various scores."""
import os as _o
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _o.environ[_v] = "1"
import sys
from multiprocessing import Pool
import numpy as np
import torch
import alib as A
import analog as N

K = int(sys.argv[1]) if len(sys.argv) > 1 else 32
D = int(sys.argv[2]) if len(sys.argv) > 2 else 64


def cap(S, Y, fx, rows):
    S = np.where(fx, -np.inf, S)[rows]; Yr = Y[rows]
    top = np.argsort(-S, 1)[:, :51]
    return float(np.take_along_axis(Yr, top, 1).sum() / np.where(fx, 0, Yr).sum())


def job(L):
    torch.set_num_threads(1)
    import lightgbm as lgb
    b = lgb.Booster(model_file=A.V2)
    lib, c = N.build_library(L, D=D)
    fx = np.zeros(256, bool); fx[A.fixed[L]] = True
    res = {}
    for corpus in ("calib-fit", "glm52-heldout"):
        d = c["d"] if corpus == "calib-fit" else A.load(corpus, L)
        sg = d["sg"]; bs = d["bsal"].astype(np.float64)
        Z = c["Z"] if corpus == "calib-fit" else N.states(bs, sg)
        Y = A.fut(bs, sg, 64)
        rows = np.isfinite(Y).all(1)
        if corpus == "calib-fit":
            rows &= c["cid"] >= N.NTRAIN_CH
        Y = np.nan_to_num(Y)
        an = lib.query(Z[rows], k=K)
        AN = np.zeros((len(Z), 768), np.float32); AN[rows] = an
        F = A.base_feats(d, L, corpus=corpus)
        v2 = A.predict(b, F, A.FEATS9)
        e64 = N.share(N.ema_rate(bs, 64, sg))
        f64, p64 = AN[:, :256], AN[:, 256:512]
        lift = f64 / np.maximum(p64, 1e-4)
        v2n = v2 / np.maximum(v2.sum(1, keepdims=True), 1e-9)
        Zp = lib.proj(Z)
        wr = N.within_request(Zp, N.share(Y if False else np.nan_to_num(A.fut(bs, sg, 64))), sg, k=8)
        wrm = np.where(np.isfinite(wr), wr, e64)
        r = dict(wr=cap(wrm, Y, fx, rows), **{f"v2+wr{a}": cap((1 - a) * v2n + a * wrm, Y, fx, rows) for a in (0.2, 0.4)},
                 **{f"e64+wr{a}": cap((1 - a) * e64 + a * wrm, Y, fx, rows) for a in (0.3, 0.5)})
        r.update(ema64=cap(e64, Y, fx, rows), an=cap(f64, Y, fx, rows), v2=cap(v2, Y, fx, rows),
                 e64xlift=cap(e64 * np.clip(lift, 0.25, 4), Y, fx, rows))
        for a in (0.2, 0.4):
            r[f"v2+an{a}"] = cap((1 - a) * v2n + a * f64, Y, fx, rows)
        res[corpus] = r
    return res


if __name__ == "__main__":
    Ls = list(range(3, 78, 6))
    with Pool(13) as p:
        R = p.map(job, Ls)
    for corpus in R[0]:
        print(corpus, K, D, {k: round(100 * np.mean([r[corpus][k] for r in R]), 2) for k in R[0][corpus]})
