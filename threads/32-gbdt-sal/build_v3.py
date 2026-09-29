#!/usr/bin/env python3
"""T32 v3 rows: router-probability features from trace2 (NQ_TRACE_PROBS capture) next to rows/CORPUS/L.npz
  build_v3.py CORPUS [NPROC]  -> rows_v3/CORPUS/L.npz  X3 [nb, ncand, 8] float32   (PRIVATE)
Per 16-token block (serve cadence), per expert, from the router's per-token output at that layer
  p   = sigmoid(logit)                       (the raw routing prob, what the gate weight is built from)
  nm  = 1 if the expert is ranked 9..16 by the selection score sigmoid + e_score_correction_bias (near miss)
  mg  = selection score - the token's 8th-best selection score (<= 0 unless selected; margin to being routed)
features (EMAs: same decays 0.5^(16/h) as ema32/ema128, reset per chain, as per-token rates):
  pema32 pema128 p16 | nmema32 nmema128 nm16 | mgema32 mg16
Also checks trace2 ids == trace ids (the recapture reproduces the routing the v1/v2 rows were built from)."""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402

import t32lib as T  # noqa: E402

FEATS_V3 = ("pema32", "pema128", "p16", "nmema32", "nmema128", "nm16", "mgema32", "mg16")
corpus = sys.argv[1]
nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 12
TR2 = os.environ.get("T32_TRACE2", f"{T.OUT}/trace2")
out = f"{T.OUT}/rows_v3/{corpus}"


def block_prob_mats(p_raw, p_selc, top16):
    Tn = p_raw.shape[0]
    nb = Tn // T.G
    bp = p_raw[: nb * T.G].astype(np.float32).reshape(nb, T.G, T.NE).mean(1)
    b = (np.arange(nb * T.G) // T.G)[:, None].repeat(8, 1)
    idx = (b * T.NE + top16[: nb * T.G, 8:16].astype(np.int64)).ravel()
    bnm = np.bincount(idx, minlength=nb * T.NE).reshape(nb, T.NE).astype(np.float32) / T.G
    ps = p_selc[: nb * T.G].astype(np.float32)
    th = np.take_along_axis(ps, top16[: nb * T.G, 7:8].astype(np.int64), 1)
    bmg = (ps - th).reshape(nb, T.G, T.NE).mean(1)
    return bp, bnm, bmg


def v3_features(bp, bnm, bmg, cand, nbc=T.CHAIN * T.SEQ // T.G):
    nb = bp.shape[0]
    a32, a128 = 0.5 ** (T.G / 32), 0.5 ** (T.G / 128)
    X = np.zeros((nb, cand.shape[1], 8), np.float32)
    for c0 in range(0, nb, nbc):
        E = np.zeros((5, T.NE))
        for k in range(c0, min(c0 + nbc, nb)):
            E[0] = E[0] * a32 + bp[k]; E[1] = E[1] * a128 + bp[k]
            E[2] = E[2] * a32 + bnm[k]; E[3] = E[3] * a128 + bnm[k]
            E[4] = E[4] * a32 + bmg[k]
            ci = cand[k].astype(np.int64)
            X[k, :, 0] = E[0][ci] * (1 - a32); X[k, :, 1] = E[1][ci] * (1 - a128); X[k, :, 2] = bp[k][ci]
            X[k, :, 3] = E[2][ci] * (1 - a32); X[k, :, 4] = E[3][ci] * (1 - a128); X[k, :, 5] = bnm[k][ci]
            X[k, :, 6] = E[4][ci] * (1 - a32); X[k, :, 7] = bmg[k][ci]
    return X


def load_probs(L):
    import glob
    import json
    parts = {}
    for f in sorted(glob.glob(f"{TR2}/windows.r*of*.json")):
        j = json.load(open(f))
        d = np.load(f"{TR2}/L{L}.r{j['rank']}of{j['world']}.npz")
        off = 0
        for name, wins in j["windows"]:
            if name == corpus:
                for k, wi in enumerate(wins):
                    s = slice(off + k * T.SEQ, off + (k + 1) * T.SEQ)
                    parts[wi] = (d["ids"][s], d["p_raw"][s], d["p_selc"][s], d["top16"][s])
            off += len(wins) * T.SEQ
    ks = sorted(parts)
    assert ks == list(range(len(ks))), (corpus, L, len(ks))
    return tuple(np.concatenate([parts[k][i] for k in ks]) for i in range(4))


def job(L):
    if os.path.exists(f"{out}/L{L}.npz"):
        return L, "exists"
    ids2, p_raw, p_selc, top16 = load_probs(L)
    ids1, _, _ = T.load_layer(L, corpus)
    same = float((np.sort(ids1, 1) == np.sort(ids2, 1)).all(1).mean())
    t16ok = float((np.sort(top16[:, :8], 1) == np.sort(ids2, 1)).all(1).mean())
    d = np.load(f"{T.OUT}/rows/{corpus}/L{L}.npz")
    X3 = v3_features(*block_prob_mats(p_raw, p_selc, top16), d["cand"])
    os.makedirs(out, exist_ok=True)
    np.savez(f"{out}/L{L}.part.npz", X3=X3)
    os.replace(f"{out}/L{L}.part.npz", f"{out}/L{L}.npz")
    return L, dict(ids_same_as_trace1=same, top16_head_eq_ids=t16ok)


if __name__ == "__main__":
    with Pool(nproc) as p:
        for r in p.imap_unordered(job, T.LAYERS):
            print(r, flush=True)
