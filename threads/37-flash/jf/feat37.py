#!/usr/bin/env python3
"""T37 jF features (PRIVATE, per layer, resumable; chain-aligned chunks keep memory per worker ~2 GB).

  feat37.py gbdt [NPROC]  -> $OUT/gbdt_rows/{train,val}/L.npz : X9 f32 [n, 9] (v2 feature order jlib37.V2F) over the
                             NON-fixed experts of every J37_GSUB-th block, y = next-64-token salience / m_L
  feat37.py jf [NPROC]    -> $OUT/feat/{split}/L.npz : X f16 [nk, 288, 22] (jlib37.INPUTS), P f32 v2 score [nk, 288],
                             y64 f16, keep i32 block index.  train: every J37_SUB-th block with a finite target;
                             val / test: all blocks (replayed).  Also $OUT/scores/v2_{val,test}/L.npy (all blocks).
m_L (target scale, T32 convention) = train-split mean of w^2 xn per routed slot of layer L.
Mixed runs (blocks37 sources): train blocks of kind prefill (teacher-forced capture trace / decode prompt rows) are
kept with the deterministic probability q of chains.json "mix" (J37_PREFILL_FRAC of the strided train rows, default
0.20), decode blocks always; val / test keep everything.  Every row file stores `kind` (0 decode / 1 prefill)."""
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402

import jlib37 as J  # noqa: E402

GSUB = int(os.environ.get("J37_GSUB", "16"))
SUB = int(os.environ.get("J37_SUB", "8"))
CH = int(os.environ.get("J37_CHUNK", "16384"))
_Q = None


def pf_q():
    global _Q
    if _Q is None:
        import json
        _Q = float(json.load(open(f"{J.OUT}/chains.json")).get("mix", {}).get("q", 1.0))
    return _Q


def train_keep(k, sub, bkind):
    """global block indices k (+ their kinds) -> keep mask for a train sampler of stride `sub` (prefill: x q)."""
    return (k % sub == 0) & ((bkind == 0) | J.pf_keep(k, pf_q()))


def gbdt_job(L):
    od = f"{J.OUT}/gbdt_rows"
    if all(os.path.exists(f"{od}/{sp}/L{L}.npz") for sp in ("train", "val")):
        return L, 0.0
    t0 = time.time()
    m = J.mL_blk(L)
    fx, _ = J.masks(L)
    nfx = np.flatnonzero(~fx)
    for sp in ("train", "val"):
        Xs, ys, ks = [], [], []
        d = J.load(sp, L)
        for s0, e0, sub in J.chunks(d, CH):
            F = J.features(sub)
            y = J.future(sub["bsal"].astype(np.float64) / m, sub["sg"], 4)
            k = np.arange(s0, e0)
            sel = train_keep(k, GSUB, sub["bkind"]) if sp == "train" else (k % GSUB == 0)
            keep = np.flatnonzero(sel & np.isfinite(y).all(1))
            if not len(keep):
                continue
            Xs.append(np.stack([F[n][keep][:, nfx] for n in J.V2F], -1).reshape(-1, 9))
            ys.append(y[keep][:, nfx].ravel())
            ks.append(np.repeat(sub["bkind"][keep], len(nfx)))
        X = np.concatenate(Xs) if Xs else np.zeros((0, 9), np.float32)
        y = np.concatenate(ys) if ys else np.zeros(0, np.float32)
        kd = np.concatenate(ks) if ks else np.zeros(0, np.int8)
        f = f"{od}/{sp}/L{L}.npz"
        np.savez(f + ".part.npz", X=X.astype(np.float32), y=y.astype(np.float32), kind=kd.astype(np.int8),
                 mL=np.float64(m))
        os.replace(f + ".part.npz", f)
    return L, time.time() - t0


def jf_job(L):
    if all(os.path.exists(f"{J.OUT}/feat/{sp}/L{L}.npz") for sp in J.SPLITS):
        return L, 0.0
    t0 = time.time()
    m = J.mL(L)
    for sp in J.SPLITS:
        d = J.load(sp, L)
        Xs, Ps, Ys, Ks, Pall, Kd = [], [], [], [], [], []
        for s0, e0, sub in J.chunks(d, CH):
            F = J.features(sub)
            y = J.future(sub["bsal"].astype(np.float64) / m, sub["sg"], 4)
            k = np.arange(s0, e0)
            keep = (np.flatnonzero(train_keep(k, SUB, sub["bkind"]) & np.isfinite(y).all(1)) if sp == "train"
                    else np.arange(e0 - s0))
            if not len(keep):
                continue
            F = {n: v[keep] for n, v in F.items()}
            P = J.v2_pred(F)                                   # v2 on the kept blocks only (all blocks off train)
            if sp != "train":
                Pall.append(P)
            Xs.append(J.net_inputs(F, P)); Ps.append(P); Ys.append(y[keep].astype(np.float16))
            Ks.append((keep + s0).astype(np.int32)); Kd.append(sub["bkind"][keep])
        cat = lambda a, shp, dt: np.concatenate(a) if a else np.zeros(shp, dt)   # noqa: E731
        f = f"{J.OUT}/feat/{sp}/L{L}.npz"
        np.savez(f + ".part.npz", X=cat(Xs, (0, J.NE, len(J.INPUTS)), np.float16), P=cat(Ps, (0, J.NE), np.float32),
                 y64=cat(Ys, (0, J.NE), np.float16), keep=cat(Ks, (0,), np.int32), kind=cat(Kd, (0,), np.int8))
        os.replace(f + ".part.npz", f)
        if sp != "train":
            pf = f"{J.OUT}/scores/v2_{sp}/L{L}.npy"
            np.save(pf + ".part.npy", cat(Pall, (0, J.NE), np.float32)); os.replace(pf + ".part.npy", pf)
    return L, time.time() - t0


if __name__ == "__main__":
    mode = sys.argv[1]
    nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    if mode == "gbdt":
        for sp in ("train", "val"):
            os.makedirs(f"{J.OUT}/gbdt_rows/{sp}", exist_ok=True)
        job = gbdt_job
    else:
        for sp in J.SPLITS:
            os.makedirs(f"{J.OUT}/feat/{sp}", exist_ok=True)
        for sp in ("val", "test"):
            os.makedirs(f"{J.OUT}/scores/v2_{sp}", exist_ok=True)
        job = jf_job
    with Pool(nproc) as p:
        for L, t in p.imap_unordered(job, J.LAYERS):
            print(f"L{L} {t:.0f}s", end=" | ", flush=True)
    print(flush=True)
