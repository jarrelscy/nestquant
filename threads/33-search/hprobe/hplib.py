"""T33c hprobe helpers: pooled (16-token block) residual / router-logit captures in corpus window order."""
import glob
import json
import os
import sys

sys.path.insert(0, "/home/coder/git/nestquant/threads/32-gbdt-sal")
import numpy as np  # noqa: E402
from scipy.signal import lfilter  # noqa: E402
import t32lib as T  # noqa: E402

HP = "/tmp/nestquant/33-search/hprobe"
POOL = f"{HP}/private/pool"
BPW = T.SEQ // T.G                                    # blocks per window
NBC = T.CHAIN * BPW                                   # blocks per chain
mL = json.load(open("/tmp/nestquant/32-gbdt-sal/models/v2_sal_tweedie1.5.txt.meta.json"))["sal_norm_mL"]
_R = None


def router():
    global _R
    if _R is None:
        _R = np.load(f"{HP}/router.npz")
    return _R


def load_pool(L, corpus, keys=("h", "lg"), d=POOL):
    parts = {}
    for f in sorted(glob.glob(f"{d}/windows.r*of*.json")):
        j = json.load(open(f))
        r, W = j["rank"], j["world"]
        z = np.load(f"{d}/pool_L{L}.r{r}of{W}.npz")
        a = {k: z[k] for k in keys}
        off = 0
        for name, wins in j["windows"]:
            for k, wi in enumerate(wins):
                if name == corpus:
                    s = slice((off + k) * BPW, (off + k + 1) * BPW)
                    parts[wi] = [a[x][s] for x in keys]
            off += len(wins)
    ks = sorted(parts)
    assert ks == list(range(len(ks))), (corpus, L, ks[:4])
    return [np.concatenate([parts[k][i] for k in ks]) for i in range(len(keys))]


def bema(M, h, nbc=NBC):
    """block EMA (half-life h tokens), per-chain reset, normalised to a mean (weights sum to 1 asymptotically)."""
    a = 0.5 ** (T.G / h)
    E = np.empty(M.shape, np.float32)
    for c0 in range(0, M.shape[0], nbc):
        E[c0:c0 + nbc] = lfilter([1.0 - a], [1.0, -a], M[c0:c0 + nbc].astype(np.float32), axis=0)
    return E


def bmean(M, k, nbc=NBC):
    """mean of the last k blocks (inclusive), truncated at chain start."""
    out = np.empty(M.shape, np.float32)
    for c0 in range(0, M.shape[0], nbc):
        X = M[c0:c0 + nbc].astype(np.float64)
        cs = np.vstack([np.zeros((1, X.shape[1])), np.cumsum(X, 0)])
        i = np.arange(X.shape[0]); lo = np.maximum(i + 1 - k, 0)
        out[c0:c0 + nbc] = (cs[i + 1] - cs[lo]) / (i + 1 - lo)[:, None]
    return out


def target(d, L):
    """next-64 normalised salience [nb, 256] by expert id (NaN where invalid)."""
    y = d["ysal"].astype(np.float64) / mL[str(L)]
    ye = np.empty_like(y); np.put_along_axis(ye, d["cand"].astype(np.int64), y, 1)
    ye[~d["valid"]] = np.nan
    return ye


def state(d):
    """v2-style normalised salience EMA32 / EMA128 [nb, 512] (build_m1.state)."""
    bc, bs = d["bcnt"].astype(np.float64), d["bsal"].astype(np.float64)

    def ema(M, h):
        a = 0.5 ** (T.G / h)
        E = np.empty(M.shape)
        for c0 in range(0, M.shape[0], NBC):
            E[c0:c0 + NBC] = lfilter([1.0], [1.0, -a], M[c0:c0 + NBC], axis=0)
        return E * ((1 - a) / T.G)
    nrm = ema(bs, 256).sum(1) / np.maximum(ema(bc, 256).sum(1), 1e-30)
    nrm = np.where(nrm > 0, nrm, 1.0)[:, None]
    return np.hstack([ema(bs, 32) / nrm, ema(bs, 128) / nrm]).astype(np.float32)


def chain_id(nb):
    return np.arange(nb) // NBC
