#!/usr/bin/env python3
"""T32 parity of the k=0 / k=6 allocation arms: T18 harness (Adapt._core_gbdt, v2 model, gmode=sync, band 0-256,
manifest k{K}_manifest.json, n_float 77-K) == offline rows_bandk0 + sim (k0.py), one chain, CPU.
  parity_k.py CORPUS L K HM"""
import json
import sys
import numpy as np
import torch
import t32lib as T
sys.path.insert(0, "/home/coder/git/nestquant/threads/18-e2e-eval")
import quantisers as Q  # noqa: E402
import lightgbm as lgb  # noqa: E402

corpus, L, K, hm = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
V2 = "/home/coder/git/nestquant/streaming/gbdt_v2sal_p64.txt"
MAN = f"{T.OUT}/k{K}_manifest.json"
m = json.load(open(MAN))
fx, start = m["default_allocation"][str(L)], m["floating_default"][str(L)]
n = T.CHAIN * T.SEQ
nb = n // T.G
ids, w, xn = (a[:n] for a in T.load_layer(L, corpus))
PH = "/tmp/nestquant/18-e2e/predecoded_H"
q = Q.Adapt(lo=f"{PH}/nq2", hi=f"{PH}/nq4", manifest=MAN, chain="1", predictor="gbdt", gmode="sync", grlo="0",
            grhi="256", gbdt_model=V2, n_float=str(77 - K), hm=str(hm))
it = torch.from_numpy(ids.astype(np.int64))
q._act = (it, torch.from_numpy(w), torch.from_numpy(xn))
_, serve_t, _ = q._core_gbdt(L, it, n)
d = np.load(f"{T.OUT}/rows_bandk0/{corpus}/L{L}.npz")
X2 = np.load(f"{T.OUT}/rows_v2_bandk0/{corpus}/L{L}.npz")["X2"]
b = lgb.Booster(model_file=V2)
pred = b.predict(np.concatenate([d["X"][:nb], X2[:nb]], -1).reshape(-1, 9))
S = T.score_blocks(pred, d["cand"][:nb], d["top"][:nb], d["e256"][:nb])
off = T.sim_layer(S, fx, start, nf=77 - K, hm=hm, lag=0)
h = serve_t[0].numpy()
print(f"k={K} hm={hm} L{L} harness==offline: {bool((h == off).all())}  mismatch blocks "
      f"{int((h != off).any(1).sum())}/{nb}", flush=True)
