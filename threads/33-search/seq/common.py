"""T33a seq: shared data/eval helpers.  Block data (bcnt, bsal) for calib-fit / glm52-heldout from T32 rows_bandall,
sm120tf from T32 private/sm120/blk.  v2 scores cached per layer.  Eval = t32lib.sim_layer semantics (sync lag 0,
hm hysteresis, band all), all-slot sal-hot, churn = new floating per refresh.  PRIVATE outputs under WD only."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/home/coder/git/nestquant/threads/32-gbdt-sal")
import t32lib as T  # noqa: E402

WD = "/tmp/nestquant/33-search/seq"
NE, G = 256, 16
LAYERS = T.LAYERS
FIXED, FDEF = T.serve_sets()
BLK = f"{T.OUT}/private/sm120/blk"
ML = {int(k): v for k, v in json.load(open(f"{T.OUT}/models/v2_sal_tweedie1.5.txt.meta.json"))["sal_norm_mL"].items()}
V2 = "/tmp/nestquant/32-gbdt-sal/models/v2_sal_tweedie1.5.txt"


def segs_of(corpus, nb=None):
    if corpus == "sm120tf":
        bs = json.load(open(f"{BLK}/sm120tf/meta.json"))["bstart"]
        return [(s, e) for s, e in zip(bs[:-1], bs[1:]) if e > s]
    nbc = T.CHAIN * T.SEQ // G
    return [(s, min(s + nbc, nb)) for s in range(0, nb, nbc)]


def load_blocks(corpus, L):
    """-> bcnt [nb,256] f32, bsal [nb,256] f64 raw"""
    if corpus == "sm120tf":
        d = np.load(f"{BLK}/sm120tf/L{L}.npz")
    else:
        d = np.load(f"{T.OUT}/rows_bandall/{corpus}/L{L}.npz")
    return d["bcnt"].astype(np.float32), d["bsal"].astype(np.float64)


def v2_scores(corpus, L):
    f = f"{WD}/v2S/{corpus}/L{L}.npy"
    if os.path.exists(f):
        return np.load(f)
    import lightgbm as lgb
    b = lgb.Booster(model_file=V2)
    if corpus == "sm120tf":
        import sm120
        d, meta = sm120.load_blk("sm120tf", L)
        F = sm120.feats(d, meta, L, set(b.feature_name()) | {"e256", "ema128"})
        fx = np.zeros(NE, bool); fx[FIXED[L]] = True
        nfx = np.flatnonzero(~fx)
        nb = d["bcnt"].shape[0]
        S = np.zeros((nb, NE), np.float32)
        for c0 in range(0, nb, 20000):
            X = np.stack([F[f][c0:c0 + 20000][:, nfx] for f in b.feature_name()], -1).reshape(-1, len(b.feature_name()))
            S[c0:c0 + 20000, nfx] = b.predict(X, num_threads=2).reshape(-1, len(nfx))
    else:
        d = np.load(f"{T.OUT}/rows_bandall/{corpus}/L{L}.npz")
        S = T.score_blocks(b.predict(T.feature_matrix(b.feature_name(), corpus, L, band="all", d=d), num_threads=2),
                           d["cand"], d["top"], d["e256"])
    os.makedirs(os.path.dirname(f), exist_ok=True)
    np.save(f, S.astype(np.float32))
    return S


from lite import replay, metrics  # noqa
