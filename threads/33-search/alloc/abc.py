#!/usr/bin/env python3
"""T33k alloc (A-D): v2, k0, band all.  Slots per layer nf, refresh interval R (16 / 8 / 4 via phase-shifted scores,
phase.py), per-layer allocation.  Per layer: sal-hot (all slots = floating, k0) and churn per refresh (within chains).
  abc.py CORPUS R TAG [NFJSON]   env NFS (grid, used when no NFJSON), HMS
  NFJSON: {L: nf} per-layer allocation (then one nf per layer).  -> $A/abc/{corpus}_R{R}_{TAG}.json"""
import os, sys, json
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from multiprocessing import Pool
import alib
from alib import T

corpus, R, TAG = sys.argv[1], int(sys.argv[2]), sys.argv[3]
NFJ = json.load(open(sys.argv[4])) if len(sys.argv) > 4 else None
NFS = [int(x) for x in os.environ.get("NFS", "64,77,90,103,128").split(",")]
HMS = [float(x) for x in os.environ.get("HMS", "0.3,0.5,0.7,1.0,1.5").split(",")]
PH = f"{alib.A}/phase/{corpus}"
NBT = alib.NBC * T.G
PHASES = list(range(0, T.G, R))


def start_list(L):
    cal = np.load(f"{alib.CACHE}/calib-fit/L{L}.npz")["bsal"].sum(0)
    base = alib.f26_ranked(L) + list(alib.fdef[L])
    rest = [e for e in np.argsort(-cal, kind="stable") if e not in set(base)]
    return base + [int(e) for e in rest]


def build(L):
    """-> S_R [nsub_total, 256] (score available at end of sub-block), sal_R [nsub_total, 256], chain segments."""
    i = T.LAYERS.index(L)
    ids, w, xn = T.load_layer(L, corpus)
    Tn = ids.shape[0]
    S0 = alib.load(corpus, L)[0]
    Ph = {o: np.load(f"{PH}/o{o}.npy", mmap_mode="r") for o in PHASES if o}
    v = (w.astype(np.float64) ** 2 * xn.astype(np.float64)[:, None])
    SR, sR, sg = [], [], []
    n0 = 0
    for ci, c0 in enumerate(range(0, Tn, NBT)):
        c1 = min(c0 + NBT, Tn); ln = c1 - c0; ns = ln // R
        tt = (np.arange(ln) // R)[:, None].repeat(8, 1)
        idx = (tt * 256 + ids[c0:c1].astype(np.int64)).ravel()
        sR.append(np.bincount(idx, weights=v[c0:c1].ravel(), minlength=ns * 256).reshape(ns, 256))
        S = np.zeros((ns, 256), np.float32)
        nb0 = ln // T.G
        cb = ci * alib.NBC
        for m in range(nb0):                       # phase 0: block m closes at 16(m+1)
            S[(T.G * (m + 1)) // R - 1] = S0[cb + m]
        for o, P in Ph.items():
            for m in range((ln - o) // T.G):       # phase o: block m closes at o + 16(m+1)
                S[(o + T.G * (m + 1)) // R - 1] = P[ci, m, i]
        SR.append(S); sg.append((n0, n0 + ns)); n0 += ns
    return np.concatenate(SR), np.concatenate(sR), sg


def job(L):
    S, s, sg = build(L)
    st = start_list(L)
    tot = s.sum()
    out = {}
    for nf in ([NFJ[str(L)]] if NFJ else NFS):
        for hm in HMS:
            sv = alib.sim_seg(S, [], st, nf, hm, sg=sg)
            out[f"{'a' if NFJ else nf}|{hm}"] = dict(sal=float((s * sv).sum() / tot), churn=alib.churn_seg(sv, sg))
    return L, out


if __name__ == "__main__":
    os.makedirs(f"{alib.A}/abc", exist_ok=True)
    with Pool(int(os.environ.get("NPROC", "18"))) as p:
        R_ = dict(p.map(job, T.LAYERS))
    json.dump(dict(corpus=corpus, R=R, per_layer={str(k): v for k, v in R_.items()}),
              open(f"{alib.A}/abc/{corpus}_R{R}_{TAG}.json", "w"))
    keys = R_[3].keys()
    for k in keys:
        print(f"R{R} {k:10s} sal {100 * np.mean([R_[L][k]['sal'] for L in R_]):6.2f} "
              f"churn {np.mean([R_[L][k]['churn'] for L in R_]):5.2f}", flush=True)
