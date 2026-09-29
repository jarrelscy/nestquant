#!/usr/bin/env python3
"""T32 end-to-end parity: T18 harness arm (quantisers.Adapt._core_gbdt, token by token, streaming/gbdt_predictor[_v2])
== offline rows + sim (t32lib) for: old model, v2 9-feature model, old x mps.  parity_v2.py CORPUS L V2MODEL"""
import sys
import numpy as np
import torch
import t32lib as T
sys.path.insert(0, "/home/coder/git/nestquant/threads/18-e2e-eval")
import quantisers as Q  # noqa: E402
import lightgbm as lgb  # noqa: E402

corpus, L, v2m = sys.argv[1], int(sys.argv[2]), sys.argv[3]
OLD = "/home/coder/git/nestquant/streaming/gbdt_p64_s5.txt"
fixed, fdef = T.serve_sets()
n = T.CHAIN * T.SEQ
ids, w, xn = (a[:n] for a in T.load_layer(L, corpus))
d = np.load(f"{T.OUT}/rows/{corpus}/L{L}.npz")
X2 = np.load(f"{T.OUT}/rows_v2/{corpus}/L{L}.npz")["X2"]
nb = n // T.G
base = dict(lo="/tmp/nestquant/18-e2e/farm_H/nq2", hi="/tmp/nestquant/18-e2e/farm_H_all4",
            manifest=T.MANIFEST, chain="1", predictor="gbdt")
for name, kw, path, scale in (("old", {}, OLD, False), ("v2", {"gbdt_model": v2m}, v2m, True),
                              ("old_x_mps", {"gbdt_scale": "mps"}, OLD, "mps")):
    q = Q.Adapt(**base, **kw)
    it = torch.from_numpy(ids.astype(np.int64))
    q._act = (it, torch.from_numpy(w), torch.from_numpy(xn))
    _, serve_t, _ = q._core_gbdt(L, it, n)
    b = lgb.Booster(model_file=path)
    X = d["X"][:nb]
    if b.num_feature() > 5:
        X = np.concatenate([X, X2[:nb]], -1)
    pred = b.predict(X.reshape(-1, b.num_feature()))
    if scale == "mps":
        pred = pred * X2[:nb, :, 3].ravel()
    S = T.score_blocks(pred, d["cand"][:nb], d["top"][:nb], d["e256"][:nb])
    off = T.sim_layer(S, fixed[L], fdef[L])
    h = serve_t[0].numpy()
    print(f"{name:10s} harness==offline: {bool((h == off).all())}  mismatch blocks {int((h != off).any(1).sum())}/{nb}",
          flush=True)
