#!/usr/bin/env python3
"""T33k alloc: candidate band at k0.  Only the top-N experts by EMA256 (serve e256, float32 recurrence per chain)
plus the current floating set (residents) are scored; the rest are not-chosen.  v2, n_float 77, sync.
  band.py CORPUS [STRIDE]   -> $A/band_{corpus}.json   (sm120tf: v2 scores recomputed via scalelib)"""
import os, sys, json
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from multiprocessing import Pool
import alib
from alib import T
sys.path.insert(0, "/home/coder/git/nestquant/threads/33-search/scale")
import scalelib as SL

corpus = sys.argv[1]; STRIDE = int(sys.argv[2]) if len(sys.argv) > 2 else 1
NS = [int(x) for x in os.environ.get("NS", "96,128,160,256").split(",")]
HMS = [float(x) for x in os.environ.get("HMS", "0.5,0.7,1.0,1.5").split(",")]
V2 = "/tmp/nestquant/32-gbdt-sal/models/v2_sal_tweedie1.5.txt"
a256 = np.float32(0.5 ** (16 / 256))


def sim_band(S, E, start, nf, hm, N, sg):
    nb = S.shape[0]
    fd = np.zeros(256, bool); fd[start[:nf]] = True
    serve = np.zeros((nb, 256), bool); f1 = np.float32(1 + hm); usz = 0
    for (s, e) in sg:
        want = fd.copy()
        for k in range(s, e):
            serve[k] = want
            allowed = want.copy()
            if N < 256:
                allowed[np.argsort(-E[k], kind="stable")[:N]] = True
            else:
                allowed[:] = True
            usz += allowed.sum()
            v = np.where(allowed, S[k], -np.inf).astype(np.float32)
            v = np.where(want, v * f1, v)
            if np.maximum(np.where(allowed, S[k], 0), 0).sum() <= 0:
                nw = want
            else:
                nw = np.zeros(256, bool); nw[np.argsort(-v, kind="stable")[:nf]] = True
            want = nw
    return serve, usz / nb


def job(L):
    if corpus in ("calib-fit", "glm52-heldout"):
        S, bs, bc = alib.load(corpus, L)
        sg = alib.chains_of(S.shape[0])
    else:
        import lightgbm as lgb
        D = SL.load(corpus, L); sg = D["sg"]; F, _ = SL.feats(D); nb = F.shape[0]
        b = lgb.Booster(model_file=V2); S = np.empty((nb, 256), np.float32)
        for c0 in range(0, nb, 8192):
            S[c0:c0 + 8192] = b.predict(F[c0:c0 + 8192].reshape(-1, 9), num_threads=1).reshape(-1, 256)
        del F
        bs, bc = D["bs"], D["bc"]
    E = np.zeros_like(S)
    c32 = bc.astype(np.float32)
    for (s, e) in sg:
        x = np.zeros(256, np.float32)
        for k in range(s, e):
            x = x * a256 + c32[k]; E[k] = x
    st = alib.f26_ranked(L) + list(alib.fdef[L])
    tot = bs.sum(); out = {}
    for N in NS:
        for hm in HMS:
            sv, u = sim_band(S, E, st, 77, hm, N, sg)
            out[f"{N}|{hm}"] = dict(sal=float((bs * sv).sum() / tot), churn=alib.churn_seg(sv, sg), rows=float(u))
    return L, out


if __name__ == "__main__":
    with Pool(int(os.environ.get("NPROC", "4"))) as p:
        R = dict(p.map(job, T.LAYERS[::STRIDE]))
    json.dump(dict(corpus=corpus, per_layer={str(k): v for k, v in R.items()}), open(f"{alib.A}/band_{corpus}.json", "w"))
    Ls = list(R)
    for key in R[Ls[0]]:
        print(f"N{key:10s} sal {100 * np.mean([R[L][key]['sal'] for L in Ls]):6.2f} churn "
              f"{np.mean([R[L][key]['churn'] for L in Ls]):5.2f} rows/layer {np.mean([R[L][key]['rows'] for L in Ls]):6.1f}",
              flush=True)
