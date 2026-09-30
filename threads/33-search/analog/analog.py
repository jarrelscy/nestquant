"""T33f analog forecasting: kNN over past layer states -> average future salience share.

State at block b (layer L): concat(sqrt(share EMA64 salience), sqrt(share EMA512 salience)) over all 256 experts
(Hellinger geometry), PCA to D dims fitted on the library.  Library = calib-fit train chains (0..23): state -> future
next-64 / next-256 salience share (+ the neighbour's present EMA64 share, for a lift feature).
Features (hit-equivalent units, x8 slots so that sum over experts ~ 8 like sema128):
  an_f64   kNN mean future next-64 share x 8
  an_p64   kNN mean present EMA64 share x 8 (so the GBDT can form the analog lift an_f64 / an_p64)
  an_f256  kNN mean future next-256 share x 8
  wr_f64   within-request: kNN among the same chain's earlier states whose next-64 is already observed
Serve-causal: states use only past routed w^2|x|^2; library futures are from other (calibration) text."""
import os as _o
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _o.environ[_v] = "1"
import os

import numpy as np
import torch

import alib as A

NE, G = 256, 16
NTRAIN_CH = 24


def ema_rate(M, h, sg):
    from scipy.signal import lfilter
    a = 0.5 ** (G / h)
    E = np.empty(M.shape, np.float64)
    for s, e in sg:
        E[s:e] = lfilter([1.0], [1.0, -a], M[s:e].astype(np.float64), axis=0)
    return E


def share(M):
    t = M.sum(1, keepdims=True)
    return (M / np.where(t > 0, t, 1.0)).astype(np.float32)


ML = [int(x) for x in os.environ.get("ML", "0").split(",")]   # multi-layer state: layer offsets (clipped to 3..77)


def states(bs, sg, hl=(64, 512)):
    return np.concatenate([np.sqrt(share(ema_rate(bs, h, sg))) for h in hl], 1)


def states_ml(corpus, L, sg, hl, bs_self=None):
    """concat of per-layer states over layers L+o, o in ML (each block scaled 1/sqrt(len(ML)))."""
    Zs = []
    for o in ML:
        Lo = min(77, max(3, L + o))
        bs = bs_self if (o == 0 and bs_self is not None) else A.load(corpus, Lo)["bsal"]
        Zs.append(states(bs.astype(np.float64), sg, hl))
    return np.concatenate(Zs, 1) / np.sqrt(len(ML))


def futures(bs, sg):
    f64 = A.fut(bs, sg, 64); f256 = A.fut(bs, sg, 256)
    ok = np.isfinite(f256).all(1)
    return share(np.nan_to_num(f64)), share(np.nan_to_num(f256)), ok


def chain_id(sg, nb):
    c = np.zeros(nb, np.int32)
    for i, (s, e) in enumerate(sg):
        c[s:e] = i
    return c


class Lib:
    def __init__(self, Z, P64, F64, F256, cid, D=64, seed=0, proj="pca"):
        self.mu = Z.mean(0)
        X = Z - self.mu
        if proj == "pca":
            C = X.T.astype(np.float64) @ X / len(X)
            w, V = np.linalg.eigh(C)
            self.W = V[:, ::-1][:, :D].astype(np.float32).copy()
        else:
            self.W = (np.random.default_rng(seed).standard_normal((Z.shape[1], D)) / np.sqrt(D)).astype(np.float32)
        self.K = torch.from_numpy(X @ self.W)
        self.kn = (self.K ** 2).sum(1)
        self.V = torch.from_numpy(np.concatenate([F64, P64, F256], 1).astype(np.float32))
        self.cid = torch.from_numpy(cid.astype(np.int64))

    def proj(self, Z):
        return torch.from_numpy((Z - self.mu) @ self.W)

    def query(self, Z, k=32, tau=None, qcid=None, chunk=4096):
        """-> [nq, 3*NE] weighted mean of library values over the k nearest (excluding same library chain)."""
        Q = self.proj(Z)
        out = np.empty((len(Z), self.V.shape[1]), np.float32)
        for c0 in range(0, len(Q), chunk):
            q = Q[c0:c0 + chunk]
            d2 = (q ** 2).sum(1, keepdim=True) - 2 * q @ self.K.T + self.kn[None]
            if qcid is not None:                       # qcid = query calib window id (-1: none); CONFLICT[qw, lw]
                qw = torch.from_numpy(qcid[c0:c0 + chunk].astype(np.int64))
                m = CONFLICT_T[qw.clamp_min(0)][:, self.cid] & (qw[:, None] >= 0)
                d2 = d2.masked_fill(m, float("inf"))
            dv, ix = torch.topk(d2, k, dim=1, largest=False)
            if tau is None:
                w = torch.full(dv.shape, 1.0 / k)
            else:
                dv = dv.clamp_min(0)
                w = torch.softmax(-(dv - dv[:, :1]) / (tau * dv[:, -1:].clamp_min(1e-12)), 1)
            out[c0:c0 + chunk] = torch.einsum("qk,qkv->qv", w, self.V[ix]).numpy()
        return out


def within_request(Zp, F64, sg, k=8, min_lag=4, min_hist=8):
    """same-chain analog: for block b, neighbours among j <= b - min_lag (j's next-64 = blocks j+1..j+4 <= b observed).
    Zp projected states (torch), F64 [nb,NE] future share.  -> [nb, NE] (NaN when fewer than min_hist candidates)."""
    nb = Zp.shape[0]
    out = np.full((nb, NE), np.nan, np.float32)
    Fv = torch.from_numpy(F64)
    for s, e in sg:
        Zc = Zp[s:e]; n = e - s
        nn2 = (Zc ** 2).sum(1)
        for c0 in range(0, n, 1024):
            c1 = min(n, c0 + 1024)
            q = Zc[c0:c1]
            hi = max(c1 - min_lag, 0)
            if hi < min_hist:
                continue
            d2 = (q ** 2).sum(1, keepdim=True) - 2 * q @ Zc[:hi].T + nn2[None, :hi]
            b = torch.arange(c0, c1)[:, None]; j = torch.arange(hi)[None]
            d2 = d2.masked_fill(j > b - min_lag, float("inf"))
            kk = min(k, hi)
            dv, ix = torch.topk(d2, kk, dim=1, largest=False)
            val = Fv[s:e][ix].mean(1)
            nav = (b[:, 0] - min_lag + 1)
            ok = (nav >= min_hist).numpy()
            out[s + c0:s + c1][ok] = val.numpy()[ok]
    return out


def _conflict():
    """[128,128] bool: calib window pairs that must not see each other in retrieval: same chain (4 windows) or
    sharing any source document (segments.npy doc_index)."""
    import json
    seg = np.load("/tmp/nestquant/corpus/glm53_calib_glmfmt_v1/c2048/segments.npy")
    rows = json.load(open("/tmp/nestquant/18-e2e/corpora/manifest.json"))["calib-fit"]["rows"]
    docs = [set(np.unique(seg[r] & (2 ** 20 - 1)).tolist()) for r in rows]
    C = np.zeros((128, 128), bool)
    for i in range(128):
        for j in range(128):
            C[i, j] = (i // 4 == j // 4) or bool(docs[i] & docs[j])
    return C


CONFLICT = _conflict()
CONFLICT_T = torch.from_numpy(CONFLICT)


def build_library(L, D=64, nch=NTRAIN_CH, proj="pca", hl=(64, 512)):
    d = A.load("calib-fit", L)
    sg = d["sg"]
    bs = d["bsal"].astype(np.float64)
    Z = states(bs, sg, hl) if ML == [0] else states_ml("calib-fit", L, sg, hl, d["bsal"])
    P64 = share(ema_rate(bs, 64, sg))
    F64, F256, ok = futures(bs, sg)
    cid = chain_id(sg, len(Z))
    wid = np.arange(len(Z)) // 128                     # calib window id (2048 tokens = 128 blocks)
    lm = ok & (cid < nch)
    lib = Lib(Z[lm], P64[lm], F64[lm], F256[lm], wid[lm], D=D, proj=proj)
    return lib, dict(d=d, Z=Z, cid=cid, wid=wid, F64=F64)
