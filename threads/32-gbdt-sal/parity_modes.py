#!/usr/bin/env python3
"""T32 parity of the new Adapt switches (gmode=sync, grlo/grhi band) vs the offline rows + sim, one chain, CPU.
  parity_modes.py CORPUS L"""
import sys
import numpy as np
import torch
import t32lib as T
sys.path.insert(0, "/home/coder/git/nestquant/threads/18-e2e-eval")
import quantisers as Q  # noqa: E402
import lightgbm as lgb  # noqa: E402

corpus, L = sys.argv[1], int(sys.argv[2])
OLD = "/home/coder/git/nestquant/streaming/gbdt_p64_s5.txt"
V2 = "/home/coder/git/nestquant/streaming/gbdt_v2sal_p64.txt"
fixed, fdef = T.serve_sets()
n = T.CHAIN * T.SEQ
nb = n // T.G
ids, w, xn = (a[:n] for a in T.load_layer(L, corpus))
base = dict(lo="/tmp/nestquant/18-e2e/farm_H/nq2", hi="/tmp/nestquant/18-e2e/farm_H_all4",
            manifest=T.MANIFEST, chain="1", predictor="gbdt")
arms = (("gbdt_sync", dict(gmode="sync"), OLD, False, "", 0),
        ("gbdt_x_mps_sync", dict(gmode="sync", gbdt_scale="mps"), OLD, True, "", 0),
        ("gbdt_x_mps_nr", dict(gbdt_scale="mps"), OLD, True, "", 1),
        ("v2_sync_bandall", dict(gmode="sync", grlo="0", grhi="256", gbdt_model=V2), V2, False, "all", 0))
for name, kw, path, mps, band, lag in arms:
    q = Q.Adapt(**base, **kw)
    it = torch.from_numpy(ids.astype(np.int64))
    q._act = (it, torch.from_numpy(w), torch.from_numpy(xn))
    _, serve_t, _ = q._core_gbdt(L, it, n)
    sfx = "_band" + band if band else ""
    d = np.load(f"{T.OUT}/rows{sfx}/{corpus}/L{L}.npz")
    X2 = np.load(f"{T.OUT}/rows_v2{sfx}/{corpus}/L{L}.npz")["X2"]
    b = lgb.Booster(model_file=path)
    X = d["X"][:nb]
    if b.num_feature() > 5:
        X = np.concatenate([X, X2[:nb]], -1)
    pred = b.predict(X.reshape(-1, b.num_feature()))
    if mps:
        pred = pred * X2[:nb, :, 3].ravel()
    S = T.score_blocks(pred, d["cand"][:nb], d["top"][:nb], d["e256"][:nb])
    off = T.sim_layer(S, fixed[L], fdef[L], lag=lag)
    h = serve_t[0].numpy()
    print(f"{name:16s} harness==offline: {bool((h == off).all())}  mismatch blocks {int((h != off).any(1).sum())}/{nb}",
          flush=True)
