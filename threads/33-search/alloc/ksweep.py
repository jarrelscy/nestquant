#!/usr/bin/env python3
"""Task 2: fixed-set size k x hysteresis margin hm, v2 sync, 77 level-4 slots = k fixed + (77-k) floating.
k fixed = top-k of the manifest fixed-26 by calib-fit salience; start order = rest of fixed-26 (by salience) then
floating_default (= T32 fixed_size.py).  Per (k, hm): all-slot sal-hot, routes-hot, churn (incl. and excl. chain
resets), sal-hot over the first 256 tokens of each chain (16 blocks) and over the rest.
  ksweep.py CORPUS -> $A/ksweep_CORPUS.json"""
import os, sys, json
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from multiprocessing import Pool
import alib as A

corpus = sys.argv[1]
KS = [int(x) for x in os.environ.get("KS", "0,6,13,20,26").split(",")]
SRC = os.environ.get("SRC", "v2")          # v2 (band all, cache/) | s5 (serve default gbdt_x_mps, native band per k)
KSEL = os.environ.get("KSEL", "manifest")   # manifest (top-k of manifest-26 by calib sal) | sal (top-k of all by calib sal)
HMS = [float(x) for x in os.environ.get("HMS", "0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.85,1,1.25,1.5,2,3").split(",")]


def job(L):
    S, bs, bc = A.load(corpus, L)
    nb = S.shape[0]
    early = (np.arange(nb) % A.NBC) < 16
    f26 = A.f26_ranked(L)
    if KSEL == "sal":        # k fixed = top-k of ALL 256 by calib-fit salience; start order = calib salience order
        cal = np.load(f"{A.CACHE}/calib-fit/L{L}.npz")["bsal"].sum(0)
        f26 = [int(e) for e in np.argsort(-cal, kind="stable")[:77]]
    out = {}
    if SRC == "s5":
        z = np.load(f"{A.A}/cache_s5/{corpus}/L{L}.npz")
    for k in KS:
        fx = f26[:k]
        if SRC == "s5":
            S = z[f"S_k{k}"]
        for hm in HMS:
            sv = A.sim(S, fx, f26[k:] + list(A.fdef[L]), 77 - k, hm)
            ch, chn = A.churn(sv), A.churn_nochain(sv)
            sv[:, fx] = True
            h = bs * sv
            out[f"{k}|{hm}"] = dict(sal=float(h.sum() / bs.sum()), cnt=float((bc * sv).sum() / bc.sum()), churn=ch,
                                     churn_nc=chn, sal_early=float(h[early].sum() / bs[early].sum()),
                                     sal_late=float(h[~early].sum() / bs[~early].sum()))
    return L, out


if __name__ == "__main__":
    with Pool(int(os.environ.get("NPROC", "18"))) as p:
        res = dict(p.map(job, A.T.LAYERS))
    summ = {}
    for key in res[3]:
        summ[key] = {m: float(np.mean([res[L][key][m] for L in A.T.LAYERS])) for m in res[3][key]}
        for bn, r in A.BANDS.items():
            summ[key][bn] = float(np.mean([res[L][key]["sal"] for L in r]))
        s = summ[key]
        print(f"k{key:10s} sal {s['sal']*100:6.2f} routes {s['cnt']*100:6.2f} churn {s['churn']:5.2f} (nc {s['churn_nc']:5.2f}) "
              f"early256 {s['sal_early']*100:6.2f} late {s['sal_late']*100:6.2f}", flush=True)
    json.dump(dict(corpus=corpus, summary=summ, per_layer=res), open(f"{A.A}/ksweep_{corpus}{'' if SRC == 'v2' else '_' + SRC}{'' if KSEL == 'manifest' else '_ksel' + KSEL}.json", "w"))
