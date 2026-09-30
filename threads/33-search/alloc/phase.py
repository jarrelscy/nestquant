#!/usr/bin/env python3
"""T33k alloc (C): phase-shifted v2 scores for refresh intervals R < 16.  The predictor keeps its 16-token block
cadence (features are exactly the trained distribution); phase o in {4,8,12} runs the streaming GBDTPredictorV2
(k0, band all) with its block boundaries shifted by o tokens (the chain's first o tokens are not fed).  Interleaving
phases gives a fresh score every R tokens.  Phase 0 = cache_scores.py cache (bitwise, see stream_parity.py).
  phase.py CORPUS O  -> $A/phase/{corpus}/o{O}.npz: S [nblk, NL, 256] f32 for blocks closing at chain_start+O+16j"""
import os, sys, time
import numpy as np
sys.path.insert(0, "/home/coder/git/nestquant/streaming")
import alib
from alib import T
from gbdt_predictor_v2 import GBDTPredictorV2

corpus, O = sys.argv[1], int(sys.argv[2])
MODEL = "/tmp/nestquant/32-gbdt-sal/models/v2_sal_tweedie1.5.txt"
LAY = T.LAYERS; NL = len(LAY)
out = f"{alib.A}/phase/{corpus}"; os.makedirs(out, exist_ok=True)
t0 = time.time()
tr = [T.load_layer(L, corpus) for L in LAY]
ids = np.stack([a[0] for a in tr], 1).astype(np.int64)
w = np.stack([a[1] for a in tr], 1).astype(np.float64)
xn = np.stack([a[2] for a in tr], 1).astype(np.float64)
del tr
Tn = ids.shape[0]
tok = T.tokens(corpus, Tn // T.SEQ)
lofs = np.arange(NL)[:, None] * 256
NBT = alib.NBC * T.G
nch = -(-Tn // NBT)
S = np.zeros((nch, alib.NBC, NL, 256), np.float32)
for ci, c0 in enumerate(range(0, Tn, NBT)):
    p = GBDTPredictorV2(LAY, {L: [] for L in LAY}, MODEL, n_float=77, hm=0.7, rlo=0, rhi=256, mode="sync",
                        num_threads=int(os.environ.get("NTHR", "8")))
    j = 0
    for t in range(c0 + O, min(c0 + NBT, Tn)):
        idx = (ids[t] + lofs).ravel()
        cnt = np.bincount(idx, minlength=NL * 256).reshape(NL, 256)
        sal = np.bincount(idx, weights=(w[t] ** 2 * xn[t][:, None]).ravel(), minlength=NL * 256).reshape(NL, 256)
        nt = [int(tok[t + 1])] if t + 1 < Tn else None
        if p.step(cnt, 1, nt, new_request=(t == c0 + O), sal=sal):
            S[ci, j] = p.S; j += 1
    p.close()
    print(f"chain {ci} {j} blocks {time.time() - t0:.0f}s", flush=True)
np.save(f"{out}/o{O}.npy", S)
print("done", corpus, O, f"{time.time() - t0:.0f}s", flush=True)
