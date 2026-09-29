#!/usr/bin/env python3
"""T32 long-horizon / domain-memory features on band-all rows -> rows_lh/CORPUS/L.npz  (PRIVATE; runs on any traced
corpus incl. prep_ids.py token-id inputs, e.g. the SM120 agent traces).  Per (block b, expert e), causal through the
end of block b, state reset per chain (= request / stream), salience normalised by norm_L as v2:
  ema8192 sema8192            hit / salience EMA rate, half-life 8192 tokens
  cum_rate cum_srate          request-cumulative hit / salience rate since chain start
  pers8 pers32 pers128        fraction of the last 8 / 32 / 128 blocks (within the chain) with >= 1 hit
  gap_mean gap_cv             mean and CV of inter-hit gaps (blocks) among the hit blocks in the last 128 blocks
                              (NaN with < 2 hits)
  dom_max dom_smax            domain memory: running max of the EMA128 hit / salience rate, decayed with half-life
                              8192 tokens (a returning domain's experts keep a score after a quiet stretch)
  dom_busy                    mean block hit rate over past busy blocks (block hits >= 2), slow-decayed (8192);
                              NaN before the first busy block
targets ysal128 ycnt128 ysal256 ycnt256 (next 128 / 256 tokens; NaN where the horizon leaves the chain)
analysis masks (not features):
  qp        [nb, 256] quiet-but-persistent at the end of block b: no hit in blocks b-1, b and pers128 >= 0.25
  ret256_sal ret1024_sal ret256_cnt ret1024_cnt [nb, 256]  per block: salience / count of "return" hits = routed hits
            whose previous hit on the same expert in the same chain is >= 256 / 1024 tokens earlier
  build_lh.py CORPUS [NPROC]"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402
from scipy.signal import lfilter  # noqa: E402

import t32lib as T  # noqa: E402

corpus = sys.argv[1]
nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 16
RB = f"{T.OUT}/rows_bandall/{corpus}"
out = f"{T.OUT}/rows_lh/{corpus}"
NBC = T.CHAIN * T.SEQ // T.G
W = 128


def ema(M, h):
    a = 0.5 ** (T.G / h)
    E = np.empty(M.shape, np.float64)
    for c0 in range(0, M.shape[0], NBC):
        E[c0:c0 + NBC] = lfilter([1.0], [1.0, -a], M[c0:c0 + NBC].astype(np.float64), axis=0)
    return E * ((1 - a) / T.G)


def per_chain(f, *Ms):
    outs = None
    for c0 in range(0, Ms[0].shape[0], NBC):
        r = f(*[M[c0:c0 + NBC] for M in Ms])
        r = r if isinstance(r, tuple) else (r,)
        if outs is None:
            outs = [np.empty((Ms[0].shape[0],) + x.shape[1:], x.dtype) for x in r]
        for o, x in zip(outs, r):
            o[c0:c0 + NBC] = x
    return outs


def pers(h, n):
    cs = np.vstack([np.zeros((1, h.shape[1])), np.cumsum(h, 0)])
    k = np.arange(h.shape[0])
    lo = np.maximum(k + 1 - n, 0)
    return (cs[k + 1] - cs[lo]) / (k + 1 - lo)[:, None]


def gaps(h):
    """h [nb, NE] bool (one chain) -> mean, cv of inter-hit block gaps among hit blocks in (b-W, b]."""
    nb, ne = h.shape
    gm = np.full((nb, ne), np.nan); gc = np.full((nb, ne), np.nan)
    k = np.arange(nb)
    for e in range(ne):
        hb = np.nonzero(h[:, e])[0]
        if len(hb) < 2:
            continue
        g = np.diff(hb).astype(np.float64)
        Q1 = np.concatenate([[0.0], np.cumsum(g)]); Q2 = np.concatenate([[0.0], np.cumsum(g * g)])
        lo = np.searchsorted(hb, k - W + 1, "left"); hi = np.searchsorted(hb, k, "right")
        n = hi - lo - 1                                     # gaps in window
        ok = n >= 1
        hc, lc = np.maximum(hi - 1, 0), np.minimum(lo, len(hb) - 1)
        s1 = Q1[hc] - Q1[lc]; s2 = Q2[hc] - Q2[lc]
        m = np.where(ok, s1 / np.maximum(n, 1), np.nan)
        var = np.maximum(np.where(ok, s2 / np.maximum(n, 1), np.nan) - m * m, 0)
        gm[:, e] = m; gc[:, e] = np.sqrt(var) / m
    return gm, gc


def dom(r, busy):
    """r [nb, NE] ema128 rate, busy [nb, NE] block-rate on busy blocks else nan (one chain)."""
    a = 0.5 ** (T.G / 8192)
    mx = np.empty_like(r); m = np.zeros(r.shape[1])
    num = np.zeros(r.shape[1]); den = np.zeros(r.shape[1]); db = np.empty_like(r)
    for k in range(r.shape[0]):
        m = np.maximum(m * a, r[k]); mx[k] = m
        b = ~np.isnan(busy[k])
        num = num * a + np.where(b, busy[k], 0); den = den * a + b
        db[k] = np.where(den > 0, num / np.maximum(den, 1e-30), np.nan)
    return mx, db


def returns(ids, v, nb):
    """token-level return events -> per block (salience, count) for gap >= 256, 1024 tokens (within chain)."""
    Tt = nb * T.G
    t = np.repeat(np.arange(Tt), 8); e = ids[:Tt].astype(np.int64).ravel(); vv = v[:Tt].ravel()
    o = np.lexsort((t, e))
    t, e, vv = t[o], e[o], vv[o]
    same = np.concatenate([[False], (e[1:] == e[:-1]) & (t[1:] // (NBC * T.G) == t[:-1] // (NBC * T.G))])
    gap = np.concatenate([[0], t[1:] - t[:-1]])
    res = []
    for G in (256, 1024):
        r = same & (gap >= G)
        S = np.zeros((nb, T.NE)); C = np.zeros((nb, T.NE))
        np.add.at(S, (t[r] // T.G, e[r]), vv[r]); np.add.at(C, (t[r] // T.G, e[r]), 1.0)
        res += [S, C]
    return res


def job(L):
    if os.path.exists(f"{out}/L{L}.npz"):
        return L, "exists"
    d = np.load(f"{RB}/L{L}.npz")
    bc, bs, cand = d["bcnt"].astype(np.float64), d["bsal"].astype(np.float64), d["cand"].astype(np.int64)
    nb = bc.shape[0]
    nrm = ema(bs, 256).sum(1) / np.maximum(ema(bc, 256).sum(1), 1e-30)
    nrm = np.where(nrm > 0, nrm, 1.0)[:, None]
    F = {"ema8192": ema(bc, 8192), "sema8192": ema(bs, 8192) / nrm}
    F["cum_rate"], F["cum_srate"] = per_chain(
        lambda c, s: (np.cumsum(c, 0) / ((np.arange(len(c)) + 1) * T.G)[:, None],
                      np.cumsum(s, 0) / ((np.arange(len(s)) + 1) * T.G)[:, None]), bc, bs)
    F["cum_srate"] = F["cum_srate"] / nrm
    h = bc > 0
    for n in (8, 32, 128):
        F[f"pers{n}"], = per_chain(lambda x: pers(x, n), h.astype(np.float64))
    F["gap_mean"], F["gap_cv"] = per_chain(gaps, h)
    r128, s128 = ema(bc, 128), ema(bs, 128) / nrm
    busy = np.where(bc >= 2, bc / T.G, np.nan)
    F["dom_max"], F["dom_busy"] = per_chain(dom, r128, busy)
    F["dom_smax"], _ = per_chain(dom, s128, busy)
    names = list(F)
    assert tuple(names) == T.FEATS_LH, names
    Fa = np.stack([np.take_along_axis(F[n], cand, 1) for n in names], -1).astype(np.float32)
    y = {}
    for n in (128, 256):
        k = n // T.G
        for nm, M in ((f"ycnt{n}", bc), (f"ysal{n}", bs)):
            Y = np.full(M.shape, np.nan)
            for c0 in range(0, nb, NBC):
                s = M[c0:c0 + NBC]
                cs = np.vstack([np.zeros((1, T.NE)), np.cumsum(s, 0)])
                kb = np.arange(max(s.shape[0] - k, 0))
                Y[c0 + kb] = cs[kb + 1 + k] - cs[kb + 1]
            y[nm] = np.take_along_axis(Y, cand, 1).astype(np.float32)
    p128 = F["pers128"]
    quiet = h.copy(); quiet[1:] |= h[:-1]
    qp = (~quiet) & (p128 >= 0.25)
    ids, w, xn = T.load_layer(L, corpus)
    v = w.astype(np.float64) ** 2 * xn.astype(np.float64)[:, None]
    r256s, r256c, r1024s, r1024c = returns(ids, v, nb)
    os.makedirs(out, exist_ok=True)
    np.savez(f"{out}/L{L}.part.npz", F=Fa, qp=qp, ret256_sal=r256s.astype(np.float32),
             ret256_cnt=r256c.astype(np.float32), ret1024_sal=r1024s.astype(np.float32),
             ret1024_cnt=r1024c.astype(np.float32), **y)
    os.replace(f"{out}/L{L}.part.npz", f"{out}/L{L}.npz")
    return L, Fa.shape


if __name__ == "__main__":
    with Pool(nproc) as p:
        for r in p.imap_unordered(job, T.LAYERS):
            print(r, flush=True)
