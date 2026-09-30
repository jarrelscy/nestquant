#!/usr/bin/env python3
"""Serve-default arm (gbdt_x_mps = streaming/gbdt_p64_s5.txt x mps128 on the native serve band: EMA256 ranks 20..120 of
the non-fixed experts scored, top 20 forced 1e3+e256), for a fixed set of size k (top-k of the manifest fixed-26 by
calib-fit salience).  Features per expert are band-independent, so they are scattered from rows_bandall (+ rows_v2_bandall
mps128) and the native band is rebuilt per k from e256 recomputed in float32 exactly like GBDTPredictor
(E = E*a + c per block, reset per chain; e256 = E*(1-a)/G).  -> cache_s5/{corpus}/L{L}.npz  S_k{k} [nb,256] f32.
  cache_s5.py CORPUS"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from multiprocessing import Pool
import alib as A
T = A.T
corpus = sys.argv[1]
S5 = "/home/coder/git/nestquant/streaming/gbdt_p64_s5.txt"
KS = [int(x) for x in os.environ.get("KS", "0,6,13,20,26").split(",")]
OUTD = f"{A.A}/cache_s5/{corpus}"


def job(L):
    import lightgbm as lgb
    f = f"{OUTD}/L{L}.npz"
    if os.path.exists(f):
        return L
    d = np.load(f"{T.OUT}/rows_bandall/{corpus}/L{L}.npz")
    X2 = np.load(f"{T.OUT}/rows_v2_bandall/{corpus}/L{L}.npz")["X2"]
    cand = d["cand"].astype(np.int64)
    nb = cand.shape[0]
    Xe = np.empty((nb, 256, 5), np.float32); np.put_along_axis(Xe, cand[..., None], d["X"], 1)
    mps = np.empty((nb, 256), np.float32); np.put_along_axis(mps, cand, X2[..., 3], 1)
    bc = d["bcnt"].astype(np.float32)
    a = np.float32(0.5 ** (T.G / 256))
    e256 = np.empty((nb, 256), np.float32)
    for c0 in range(0, nb, A.NBC):
        E = np.zeros(256, np.float32)
        for k in range(c0, min(c0 + A.NBC, nb)):
            E = E * a + bc[k]
            e256[k] = E * ((1 - a) / T.G)
    b = lgb.Booster(model_file=S5)
    f26 = A.f26_ranked(L)
    out = {}
    for k in KS:
        fx = np.zeros(256, bool); fx[f26[:k]] = True
        order = np.argsort(-np.where(fx, -np.inf, e256), 1, kind="stable")
        c, top = order[:, 20:121], order[:, :20]
        Xc = np.take_along_axis(Xe, c[..., None], 1).reshape(-1, 5)
        pr = b.predict(Xc, num_threads=1).reshape(c.shape) * np.take_along_axis(mps, c, 1)
        S = np.zeros((nb, 256), np.float32)
        np.put_along_axis(S, c, pr.astype(np.float32), 1)
        np.put_along_axis(S, top, (1e3 + np.take_along_axis(e256, top, 1)).astype(np.float32), 1)
        out[f"S_k{k}"] = S
        if k == 26:   # consistency: band-all cand ordering == our e256 order on the non-fixed experts
            nfix = 256 - k
            out["order_match"] = np.float64((order[:, :nfix] == cand[:, :nfix]).mean())
    np.savez(f + ".tmp.npz", **out)
    os.replace(f + ".tmp.npz", f)
    return L


if __name__ == "__main__":
    os.makedirs(OUTD, exist_ok=True)
    with Pool(int(os.environ.get("NPROC", "16"))) as p:
        for L in p.imap_unordered(job, T.LAYERS):
            pass
    print("done", corpus)
