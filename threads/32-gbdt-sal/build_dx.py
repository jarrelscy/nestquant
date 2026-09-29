#!/usr/bin/env python3
"""T32 cross-layer DRIFT features on band-all rows -> rows_dx/CORPUS/L.npz  F [nb, 256(cand), 3]  (PRIVATE).
Causal at the end of block b (sync: every layer has routed the whole block), chain reset:
  sema64_J    normalised salience EMA (half-life 64 tokens) of layer J (norm = layer's causal salience per hit, as v2)
  D_J(b)      sema64_J(b) - sema64_J(b-4)  (change over the last 64 tokens; 0 for b < 4 in the chain), kept only on
              the top-32 experts of J by sema64 at b or at b-4 (risers and fallers), 0 elsewhere
  C_{J->L}    [256, 256] P(e at L | j at J) on the same token, calib-fit routing
  dx_lo = mean_{J in L-2, L-1} D_J @ C_{J->L}      dx_hi = same for J in L+1, L+2      (NaN if no such layer)
  dx_self = D_L                                    (own-layer drift, reference)
  build_dx.py CORPUS [NPROC]"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402
from scipy.signal import lfilter  # noqa: E402

import t32lib as T  # noqa: E402

corpus = sys.argv[1]
nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 16
out = f"{T.OUT}/rows_dx/{corpus}"
NBC = T.CHAIN * T.SEQ // T.G
TOPK, LAGB = 32, 4
FEATS_DX = ("dx_lo", "dx_hi", "dx_self")


def ema(M, h):
    a = 0.5 ** (T.G / h)
    E = np.empty(M.shape, np.float64)
    for c0 in range(0, M.shape[0], NBC):
        E[c0:c0 + NBC] = lfilter([1.0], [1.0, -a], M[c0:c0 + NBC].astype(np.float64), axis=0)
    return E * ((1 - a) / T.G)


def drift(J):
    d = np.load(f"{T.OUT}/rows_bandall/{corpus}/L{J}.npz")
    bc, bs = d["bcnt"].astype(np.float64), d["bsal"].astype(np.float64)
    nrm = ema(bs, 256).sum(1) / np.maximum(ema(bc, 256).sum(1), 1e-30)
    s = ema(bs, 64) / np.where(nrm > 0, nrm, 1.0)[:, None]
    prev = np.zeros_like(s)
    for c0 in range(0, s.shape[0], NBC):
        n = min(NBC, s.shape[0] - c0)
        prev[c0 + LAGB:c0 + n] = s[c0:c0 + n - LAGB]
        prev[c0:c0 + LAGB] = s[c0:c0 + LAGB]
    D = s - prev
    keep = np.zeros(s.shape, bool)
    for M in (s, prev):
        np.put_along_axis(keep, np.argsort(-M, 1)[:, :TOPK], True, 1)
    return np.where(keep, D, 0.0)


def coact(J, L):
    ij, _, _ = T.load_layer(J, "calib-fit")
    il, _, _ = T.load_layer(L, "calib-fit")
    n = ij.shape[0]
    A = np.zeros((n, T.NE), np.float32); np.put_along_axis(A, ij.astype(np.int64), 1.0, 1)
    B = np.zeros((n, T.NE), np.float32); np.put_along_axis(B, il.astype(np.int64), 1.0, 1)
    C = (A.T @ B).astype(np.float64)
    return C / np.maximum(A.sum(0)[:, None], 1.0)


def job(L):
    if os.path.exists(f"{out}/L{L}.npz"):
        return L, "exists"
    cand = np.load(f"{T.OUT}/rows_bandall/{corpus}/L{L}.npz")["cand"].astype(np.int64)
    F = {}
    for name, Js in (("dx_lo", (L - 2, L - 1)), ("dx_hi", (L + 1, L + 2))):
        Js = [J for J in Js if J in T.LAYERS]
        F[name] = (np.mean([drift(J) @ coact(J, L) for J in Js], 0) if Js
                   else np.full(cand.shape, np.nan))
    F["dx_self"] = drift(L)
    Fa = np.stack([np.take_along_axis(F[n], cand, 1) for n in FEATS_DX], -1).astype(np.float32)
    os.makedirs(out, exist_ok=True)
    np.savez(f"{out}/L{L}.part.npz", F=Fa)
    os.replace(f"{out}/L{L}.part.npz", f"{out}/L{L}.npz")
    return L, Fa.shape


if __name__ == "__main__":
    with Pool(nproc) as p:
        for r in p.imap_unordered(job, T.LAYERS):
            print(r, flush=True)
