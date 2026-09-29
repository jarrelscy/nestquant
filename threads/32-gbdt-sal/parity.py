#!/usr/bin/env python3
"""T32 parity: block-fed features + vectorised sim (t32lib) == token-by-token GBDTPredictor drive of
quantisers.Adapt._core_gbdt (next_refresh, hysteresis) on one traced chain.  parity.py CORPUS L [MODEL]"""
import sys

import numpy as np

import t32lib as T
from gbdt_predictor import GBDTPredictor, G

corpus, L = sys.argv[1], int(sys.argv[2])
model = sys.argv[3] if len(sys.argv) > 3 else None
fixed, fdef = T.serve_sets()
ids, w, xn = T.load_layer(L, corpus)
n = T.CHAIN * T.SEQ
ids = ids[:n]
tok = T.tokens(corpus, T.CHAIN)
seg = T.seg_of(tok)
cnt, cnta, nans, sal, sl = T.block_mats(ids, w[:n], xn[:n], seg)
X, cand, top, e = T.chain_features(L, fixed, cnt, cnta, nans, sl)
P0 = GBDTPredictor([L], {L: fixed[L]}, model_path=model, mode="sync")
S = T.score_blocks(P0.bst.predict(X.reshape(-1, 5)), cand, top, e)
serve = T.sim_layer(S, fixed[L], fdef[L])
# reference: token-by-token, exactly as _core_gbdt (+ serve-style token ids)
P = GBDTPredictor([L], {L: fixed[L]}, model_path=model, mode="next_refresh", num_threads=4)
fx = np.zeros(256, bool); fx[fixed[L]] = True
fd = np.zeros(256, bool); fd[[x for x in fdef[L] if x not in set(fixed[L])][:51]] = True
want = fd.copy()
ref = np.zeros_like(serve)
feats = []
for t in range(n):
    if t % G == 0:
        ref[t // G] = want
    c = np.bincount(ids[t], minlength=256).astype(np.float64)[None]
    nt = [int(tok[t + 1])] if t + 1 < len(tok) else None
    if P.step(c, 1, nt, t == 0):
        wv = P.target(want[None])
        if wv is not None:
            want = wv[0] & ~fx
P.close()
print("serve sets equal:", bool((ref == serve).all()), "mismatch blocks:", int((ref != serve).any(1).sum()), "/", len(ref))
