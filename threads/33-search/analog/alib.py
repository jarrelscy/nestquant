"""T33f analog: shared data/eval helpers (own eval path; features via T32 sm120.feats, sim via sm120.replay)."""
import os as _o
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _o.environ[_v] = "1"
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, "/home/coder/git/nestquant/threads/32-gbdt-sal")
import numpy as np  # noqa: E402
import t32lib as T  # noqa: E402
import sm120 as S1  # noqa: E402

NE, G = 256, 16
OUT = "/tmp/nestquant/33-search/analog"
BLK = "/tmp/nestquant/32-gbdt-sal/private/sm120/blk"
V2 = "/tmp/nestquant/32-gbdt-sal/models/v2_sal_tweedie1.5.txt"
FEATS9 = ["ema32", "ema128", "mem_cur_state", "tok_since_hit", "hits16", "sema32", "sema128", "sal16", "mps128"]
fixed, fdef = T.serve_sets()
LAYERS = T.LAYERS
mL = {int(k): v for k, v in json.load(open(V2 + ".meta.json"))["sal_norm_mL"].items()}


def load(corpus, L):
    """-> dict bcnt bcnta nans segl bsal (np arrays) + sg list of (s,e) chain block ranges."""
    z = np.load(f"{BLK}/{corpus}/L{L}.npz")
    d = {k: z[k] for k in ("bcnt", "bcnta", "nans", "segl")}
    if "bsal" in z.files:
        d["bsal"] = z["bsal"]
    else:
        d["bsal"] = np.load(f"/tmp/nestquant/32-gbdt-sal/rows_bandall/{corpus}/L{L}.npz")["bsal"]
    meta = json.load(open(f"{BLK}/{corpus}/meta.json"))
    bs = list(meta["bstart"])
    nb = d["bcnt"].shape[0]
    if bs[-1] < nb:                      # calib/heldout meta drops the trailing partial chain
        bs.append(nb)
    d["sg"] = [(a, b) for a, b in zip(bs[:-1], bs[1:]) if b > a]
    return d


class _D(dict):
    files = ("bsal",)


def base_feats(d, L, need=None, corpus=None):
    """9 v2 features [nb, NE] in expert order.  calib/heldout: T32's serve-exact rows (GBDTPredictor code path,
    bitwise = serve); other streams: sm120.feats (vectorised; float-rounding differences only)."""
    if corpus in ("calib-fit", "glm52-heldout") and (need is None or set(need) <= set(FEATS9)):
        r = np.load(f"/tmp/nestquant/32-gbdt-sal/rows_bandall/{corpus}/L{L}.npz")
        X = np.concatenate([r["X"], np.load(f"/tmp/nestquant/32-gbdt-sal/rows_v2_bandall/{corpus}/L{L}.npz")["X2"]], -1)
        inv = np.argsort(r["cand"].astype(np.int64), 1)
        X = np.take_along_axis(X, inv[..., None], 1)
        return {n: np.ascontiguousarray(X[..., i]) for i, n in enumerate(FEATS9)}
    need = set(need or FEATS9) | {"e256", "ema128"}
    dd = _D(d)
    meta = {"bstart": [s for s, _ in d["sg"]] + [d["sg"][-1][1]]}
    return S1.feats(dd, meta, L, need)


def fut(M, sg, h):
    """future sum over the next h/G blocks after block b (b+1..b+h/G), NaN where horizon leaves the chain."""
    k = h // G
    F = np.full(M.shape, np.nan, np.float32)
    for s, e in sg:
        cs = np.vstack([np.zeros((1, M.shape[1])), np.cumsum(M[s:e].astype(np.float64), 0)])
        n = e - s
        b = np.arange(max(n - k, 0))
        F[s + b] = cs[b + 1 + k] - cs[b + 1]
    return F


def sim(S, L, sg, hm=0.5):
    fx = np.zeros(NE, bool); fx[fixed[L]] = True
    fd = np.zeros(NE, bool); fd[[e for e in fdef[L] if e not in set(fixed[L])][:51]] = True
    return S1.replay(S.astype(np.float32), fx, fd, sg, hm=hm), fx


def metrics(sv, fx, bs, sg):
    """sal-hot (all slots), churn (hot_eval style over all consecutive blocks) and churn within chains."""
    ch_all = float((sv[1:] & ~sv[:-1]).sum(1).mean())
    c = (sv[1:] & ~sv[:-1]).sum(1).astype(np.float64)
    num = sum(c[s:e - 1].sum() for s, e in sg); den = sum(max(e - s - 1, 0) for s, e in sg)
    hot = sv | fx
    return dict(sal=float((bs * hot).sum() / bs.sum()), churn=ch_all, churn_in=float(num / den),
                num=float((bs * hot).sum()), den=float(bs.sum()))


def predict(booster, F, names, nthr=1):
    X = np.stack([F[n] for n in names], -1).reshape(-1, len(names)).astype(np.float32)
    nb = F[names[0]].shape[0]
    return booster.predict(X, num_threads=nthr).reshape(nb, NE).astype(np.float32)
