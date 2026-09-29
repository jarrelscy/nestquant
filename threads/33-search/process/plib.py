"""T33h (process): shared loaders / replay / metric for stochastic-process predictor features.
Expert-indexed per-block matrices [nb, 256]; chains (predictor state resets) given as segs [(s, e)].
Corpora: calib-fit / glm52-heldout (rows_bandall + rows_v2_bandall, 512-block chains), sm120tf (private blk, 7 chains).
PRIVATE inputs read-only from /tmp/nestquant/32-gbdt-sal; outputs under /tmp/nestquant/33-search/process."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/home/coder/git/nestquant/threads/32-gbdt-sal")
import t32lib as T  # noqa: E402

NE, G = 256, 16
OUT = "/tmp/nestquant/33-search/process"
V2 = "/tmp/nestquant/32-gbdt-sal/models/v2_sal_tweedie1.5.txt"
V2F = list(T.FEATS5) + list(T.FEATS_V2)
BLK = "/tmp/nestquant/32-gbdt-sal/private/sm120/blk"
fixed, fdef = T.serve_sets()


def load(corpus, L):
    """-> dict F (9 v2 feats, name -> [nb, NE] f32), bcnt, bsal [nb, NE] f64, segs, mL (per-layer w^2|x|^2 per slot)."""
    if corpus in ("calib-fit", "glm52-heldout"):
        d = np.load(f"{T.OUT}/rows_bandall/{corpus}/L{L}.npz")
        X2 = np.load(f"{T.OUT}/rows_v2_bandall/{corpus}/L{L}.npz")["X2"]
        cand = d["cand"].astype(np.int64)
        nb = cand.shape[0]
        F = {}
        for i, n in enumerate(T.FEATS5):
            M = np.zeros((nb, NE), np.float32); np.put_along_axis(M, cand, d["X"][..., i], 1); F[n] = M
        for i, n in enumerate(T.FEATS_V2):
            M = np.zeros((nb, NE), np.float32); np.put_along_axis(M, cand, X2[..., i], 1); F[n] = M
        nbc = T.CHAIN * T.SEQ // G
        segs = [(s, min(s + nbc, nb)) for s in range(0, nb, nbc)]
        return dict(F=F, bcnt=d["bcnt"].astype(np.float64), bsal=d["bsal"].astype(np.float64), segs=segs,
                    mL=float(d["slot_sal_sum"] / d["slots"]))
    import sm120 as S  # noqa: E402
    d, meta = S.load_blk(corpus, L)
    F = S.feats(d, meta, L, set(V2F) | {"e256"})
    F.pop("e256", None)
    segs = [(s, e) for s, e in S.segs(meta) if e > s]
    bs = d["bsal"].astype(np.float64)
    return dict(F=F, bcnt=d["bcnt"].astype(np.float64), bsal=bs, segs=segs,
                mL=float(bs.sum() / (d["bcnt"].astype(np.float64).sum())))


def target(M, segs, k=4):
    """next-k-block sum (blocks b+1..b+k) within chain; NaN where the horizon leaves the chain."""
    Y = np.full(M.shape, np.nan, np.float32)
    for s, e in segs:
        cs = np.vstack([np.zeros((1, M.shape[1])), np.cumsum(M[s:e], 0)])
        b = np.arange(max(e - s - k, 0))
        Y[s + b] = cs[b + 1 + k] - cs[b + 1]
    return Y


def replay(S, L, segs, hm=0.5, nf=51):
    """= t32lib.sim_layer lag 0 (sync), chains = segs. -> serve [nb, NE] bool (floating set serving block k)."""
    fx = np.zeros(NE, bool); fx[fixed[L]] = True
    fd = np.zeros(NE, bool); fd[[e for e in fdef[L] if e not in set(fixed[L])][:nf]] = True
    nb = S.shape[0]
    serve = np.zeros((nb, NE), bool)
    V = np.where(fx, -np.inf, S).astype(np.float32)
    pos = np.where(fx, 0, np.maximum(S, 0)).sum(1) > 0
    f = np.float32(1 + hm)
    for s, e in segs:
        want = fd.copy()
        for k in range(s, e):
            serve[k] = want
            if pos[k]:
                v = np.where(want, V[k] * f, V[k])
                nw = np.zeros(NE, bool); nw[np.argsort(-v, kind="stable")[:nf]] = True
                want = nw
    return serve


def metric(sv, L, bsal, bcnt, segs):
    fx = np.zeros(NE, bool); fx[fixed[L]] = True
    ch_num = ch_den = 0.0
    for s, e in segs:
        ch_num += float((sv[s + 1:e] & ~sv[s:e - 1]).sum()); ch_den += max(e - s - 1, 0)
    hot = sv | fx
    return dict(churn_g=float((sv[1:] & ~sv[:-1]).sum(1).mean()), sal=float((bsal * hot).sum() / bsal.sum()), cnt=float((bcnt * hot).sum() / bcnt.sum()),
                churn=ch_num / max(ch_den, 1))


def gbdt_scores(booster, F, extra=None):
    names = booster.feature_name()
    src = dict(F); src.update(extra or {})
    nb = next(iter(F.values())).shape[0]
    X = np.stack([src[n] for n in names], -1).reshape(-1, len(names))
    return booster.predict(X, num_threads=1).reshape(nb, NE).astype(np.float32)
