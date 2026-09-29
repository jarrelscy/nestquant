#!/usr/bin/env python3
"""T32 token -> expert affinity features (T1) + token lookahead (T2) on band-all rows -> rows_t1/CORPUS/L.npz
(PRIVATE: derived per token).  Tables fit on calib-fit only; for calib-fit rows the table is out-of-fold (2 folds by
chain parity) so train features never see their own targets.  Salience v_te = w_te^2 |x_t|^2 / m_L (hit units).
  A0[tok][e]   mean salience of expert e at a position whose input token is tok       (Bayes-shrunk to the layer
  Af[tok][e]   mean salience of e summed over the next 16 positions after tok           prior, alpha = ALPHA obs)
  A2[bigram]   A0 keyed by hash(prev tok, tok)
Features per (block b, expert) at the end of block b (causal = serve-available, chain reset):
  t1_e16 t1_e64    token-EMA (half-life 16 / 64 tokens) of A0[tok_t]
  t1b_e16          token-EMA16 of A2[bigram_t]
  t1f_e16 t1f_last token-EMA16 of Af[tok_t]; Af of the block's last token     (causal lookahead: bigram-LM-like)
  t2_n16 t2_n64    sum of A0 over the TRUE next 16 / 64 tokens (UPPER BOUND, not serve-available; NaN past chain end)
  build_t1.py CORPUS [NPROC]"""
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402
from scipy.signal import lfilter  # noqa: E402

import t32lib as T  # noqa: E402

corpus = sys.argv[1]
nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 16
out = f"{T.OUT}/rows_t1/{corpus}"
CT = T.CHAIN * T.SEQ
ALPHA = 4.0
FEATS_T1 = ("t1_e16", "t1_e64", "t1b_e16", "t1f_e16", "t1f_last", "t2_n16", "t2_n64")
mL = json.load(open("/tmp/nestquant/32-gbdt-sal/models/v2_sal_tweedie1.5.txt.meta.json"))["sal_norm_mL"]


def dense(ids, w, xn, L):
    Tn = ids.shape[0]
    v = (w.astype(np.float64) ** 2 * xn.astype(np.float64)[:, None]) / mL[str(L)]
    P = np.zeros((Tn, T.NE), np.float32)
    np.put_along_axis(P, ids.astype(np.int64), v.astype(np.float32), 1)     # top-8 ids distinct per token
    return P


def fwd(P, n):
    """sum over positions t+1..t+n within the chain (truncated)."""
    out_ = np.zeros(P.shape, np.float32)
    for c0 in range(0, P.shape[0], CT):
        s = P[c0:c0 + CT].astype(np.float64)
        cs = np.vstack([np.zeros((1, T.NE)), np.cumsum(s, 0)])
        k = np.arange(s.shape[0])
        out_[c0:c0 + CT] = cs[np.minimum(k + 1 + n, s.shape[0])] - cs[k + 1]
    return out_


def bigram(tok):
    prev = np.concatenate([[-1], tok[:-1]])
    prev[::CT] = -1
    return (prev + 1) * 1000003 + tok


def fit(keys, P):
    u, inv = np.unique(keys, return_inverse=True)
    S = np.zeros((len(u), T.NE)); np.add.at(S, inv, P)
    n = np.bincount(inv, minlength=len(u)).astype(np.float64)
    prior = P.mean(0, dtype=np.float64)
    return u, ((S + ALPHA * prior) / (n + ALPHA)[:, None]).astype(np.float32), prior.astype(np.float32)


def lookup(tab, keys):
    u, A, prior = tab
    i = np.searchsorted(u, keys); i = np.minimum(i, len(u) - 1)
    hit = u[i] == keys
    return np.where(hit[:, None], A[i], prior[None])


def tema(X, h, ends):
    a = 0.5 ** (1.0 / h)
    o = np.empty((len(ends), T.NE), np.float32)
    E = np.empty(X.shape, np.float64)
    for c0 in range(0, X.shape[0], CT):
        E[c0:c0 + CT] = lfilter([1.0 - a], [1.0, -a], X[c0:c0 + CT].astype(np.float64), axis=0)
    o[:] = E[ends]
    return o


def job(L):
    if os.path.exists(f"{out}/L{L}.npz"):
        return L, "exists"
    ncal = 16384 * T.G // T.SEQ
    idc, wc, xc = T.load_layer(L, "calib-fit")
    Pc = dense(idc, wc, xc, L); Fc = fwd(Pc, 16)
    tc = T.tokens("calib-fit", ncal)
    if corpus == "calib-fit":
        tok, P = tc, Pc
        chain = np.arange(len(tc)) // CT
        A0 = np.empty(P.shape, np.float32); Af = np.empty(P.shape, np.float32); A2 = np.empty(P.shape, np.float32)
        for f in (0, 1):
            tr, te = chain % 2 != f, chain % 2 == f
            A0[te] = lookup(fit(tc[tr], Pc[tr]), tc[te])
            Af[te] = lookup(fit(tc[tr], Fc[tr]), tc[te])
            bg = bigram(tc)
            A2[te] = lookup(fit(bg[tr], Pc[tr]), bg[te])
    else:
        d = np.load(f"{T.OUT}/rows_bandall/{corpus}/L{L}.npz")
        nb = d["bcnt"].shape[0]
        tok = T.tokens(corpus, nb * T.G // T.SEQ)
        A0 = lookup(fit(tc, Pc), tok); Af = lookup(fit(tc, Fc), tok); A2 = lookup(fit(bigram(tc), Pc), bigram(tok))
    del Pc, Fc
    nb = len(tok) // T.G
    ends = np.arange(nb) * T.G + T.G - 1
    F = {"t1_e16": tema(A0, 16, ends), "t1_e64": tema(A0, 64, ends), "t1b_e16": tema(A2, 16, ends),
         "t1f_e16": tema(Af, 16, ends), "t1f_last": Af[ends]}
    for n in (16, 64):
        f = fwd(A0, n)[ends]
        pos = ends % CT
        f[pos + n >= CT] = np.nan
        F[f"t2_n{n}"] = f
    cand = np.load(f"{T.OUT}/rows_bandall/{corpus}/L{L}.npz")["cand"].astype(np.int64)
    assert cand.shape[0] == nb
    Fa = np.stack([np.take_along_axis(F[n], cand, 1) for n in FEATS_T1], -1).astype(np.float32)
    os.makedirs(out, exist_ok=True)
    np.savez(f"{out}/L{L}.part.npz", F=Fa)
    os.replace(f"{out}/L{L}.part.npz", f"{out}/L{L}.npz")
    return L, Fa.shape


if __name__ == "__main__":
    with Pool(nproc) as p:
        for r in p.imap_unordered(job, T.LAYERS):
            print(r, flush=True)
