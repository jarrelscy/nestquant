#!/usr/bin/env python3
"""T32 extra features on band-all rows (every expert a candidate) -> rows_x/CORPUS/L.npz  F [nb, 256(cand), K],
ysal32 / ycnt32 [nb, 256(cand)].  PRIVATE.  Per 16-token block b, causal (through token 16(b+1)-1) unless noted,
EMA state reset per chain (= request), rates per token, salience normalised by the layer's causal salience per hit
(norm_L = sum EMA256 sal / sum EMA256 hits, as v2):
  cx_e32 cx_e128 cx_s32 cx_s128   cross-layer: (EMA32/128 hit rate | norm. salience rate of layer L-1) @ C[L]
                                  C[j,e] = P(e at L | j at L-1) from calib-fit tokens (stats_x.py); NaN at L3
  co_e32 co_s32                   within-layer: sum_j A[i,j] * (ema32_j | sema32_j), A[i,j] = P(i | j same token)
  ema512 ema2048 sema512 sema2048 long EMAs (hits / normalised salience)
  tc_word tc_num tc_code tc_punct tc_ws pos   token-class shares of the block's 16 input tokens (tokclass.py);
                                  pos = tokens since request (chain) start
  mtp{k}_cnt mtp{k}_sal (k=1,2,4) ORACLE: true routing of the next k tokens 16(b+1)..16(b+1)+k-1 (MTP upper bound)
targets: ysal32 / ycnt32 over the next 32 tokens (H=32 horizon)."""
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
out = f"{T.OUT}/rows_x/{corpus}"
NBC = T.CHAIN * T.SEQ // T.G
KS = (1, 2, 4)


def ema(M, h):
    """per chain: E_k = a E_{k-1} + M_k, returned as a per-token rate E (1-a)/G."""
    a = 0.5 ** (T.G / h)
    E = np.empty(M.shape, np.float64)
    for c0 in range(0, M.shape[0], NBC):
        E[c0:c0 + NBC] = lfilter([1.0], [1.0, -a], M[c0:c0 + NBC].astype(np.float64), axis=0)
    return E * ((1 - a) / T.G)


def norm_of(bc, bs):
    n = ema(bs, 256).sum(1) / np.maximum(ema(bc, 256).sum(1), 1e-30)
    return np.where(n > 0, n, 1.0)[:, None]


def job(L):
    if os.path.exists(f"{out}/L{L}.npz"):
        return L, "exists"
    st = np.load(f"{T.OUT}/stats/coact.npz")
    d = np.load(f"{RB}/L{L}.npz")
    bc, bs, cand = d["bcnt"].astype(np.float64), d["bsal"].astype(np.float64), d["cand"].astype(np.int64)
    nb = bc.shape[0]
    nrm = norm_of(bc, bs)
    F = {}
    if L - 1 in T.LAYERS:
        p = np.load(f"{RB}/L{L - 1}.npz")
        pc, ps = p["bcnt"].astype(np.float64), p["bsal"].astype(np.float64)
        pn = norm_of(pc, ps)
        C = st[f"C{L}"].astype(np.float64)
        F["cx_e32"], F["cx_e128"] = ema(pc, 32) @ C, ema(pc, 128) @ C
        F["cx_s32"], F["cx_s128"] = (ema(ps, 32) / pn) @ C, (ema(ps, 128) / pn) @ C
    else:
        for n in ("cx_e32", "cx_e128", "cx_s32", "cx_s128"):
            F[n] = np.full((nb, T.NE), np.nan)
    A = st[f"A{L}"].astype(np.float64)
    e32, s32 = ema(bc, 32), ema(bs, 32) / nrm
    F["co_e32"], F["co_s32"] = e32 @ A.T, s32 @ A.T
    F["ema512"], F["ema2048"] = ema(bc, 512), ema(bc, 2048)
    F["sema512"], F["sema2048"] = ema(bs, 512) / nrm, ema(bs, 2048) / nrm
    tok = T.tokens(corpus, nb * T.G // T.SEQ)
    tcl = np.load(f"{T.OUT}/stats/tokclass.npy")[tok[: nb * T.G]].reshape(nb, T.G)
    for i, n in enumerate(("tc_word", "tc_num", "tc_code", "tc_punct", "tc_ws")):
        F[n] = np.repeat((tcl == i).mean(1, keepdims=True), T.NE, 1)
    F["pos"] = np.repeat((((np.arange(nb) % NBC) + 1) * T.G).astype(np.float64)[:, None], T.NE, 1)
    ids, w, xn = T.load_layer(L, corpus)
    v = w.astype(np.float64) ** 2 * xn.astype(np.float64)[:, None]
    for k in KS:
        mc, msl = np.zeros((nb, T.NE)), np.zeros((nb, T.NE))
        for j in range(k):
            t = (np.arange(nb) + 1) * T.G + j
            ok = ((np.arange(nb) % NBC) < NBC - 1) & (t < ids.shape[0])
            b = np.nonzero(ok)[0]
            np.add.at(mc, (np.repeat(b, 8), ids[t[b]].astype(np.int64).ravel()), 1.0)
            np.add.at(msl, (np.repeat(b, 8), ids[t[b]].astype(np.int64).ravel()), v[t[b]].ravel())
        F[f"mtp{k}_cnt"], F[f"mtp{k}_sal"] = mc, msl / nrm
    names = list(F)
    assert tuple(names) == T.FEATS_X, names
    Fa = np.stack([np.take_along_axis(F[n], cand, 1) for n in names], -1).astype(np.float32)
    y32 = {}
    for nm, M in (("ycnt32", bc), ("ysal32", bs)):
        Y = np.full(M.shape, np.nan)
        for c0 in range(0, nb, NBC):
            s = M[c0:c0 + NBC]
            cs = np.vstack([np.zeros((1, T.NE)), np.cumsum(s, 0)])
            kb = np.arange(s.shape[0] - 2)
            Y[c0 + kb] = cs[kb + 3] - cs[kb + 1]
        y32[nm] = np.take_along_axis(Y, cand, 1).astype(np.float32)
    os.makedirs(out, exist_ok=True)
    np.savez(f"{out}/L{L}.part.npz", F=Fa, **y32)
    os.replace(f"{out}/L{L}.part.npz", f"{out}/L{L}.npz")
    return L, Fa.shape


if __name__ == "__main__":
    with Pool(nproc) as p:
        for r in p.imap_unordered(job, T.LAYERS):
            print(r, flush=True)
