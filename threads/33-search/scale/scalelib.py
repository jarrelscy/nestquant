"""T33j (scale): vectorised v2 feature/eval path over block streams (calib-fit, glm52-heldout, sm120tf).
Features = streaming/gbdt_predictor.py (5 count feats) + t32lib.v2_features (4 salience feats), computed for ALL
256 experts per block with lfilter, per chain.  mem_cur_state either 'correct' (think/answer from token ids) or
'stuck' (serve today: no ids -> always think -> EMA2048 of all counts).
Eval = t32lib.sim_layer (lag 0, hm 0.5, nf 51) and decomp.replay (cap) semantics; per-block covered / total salience is
returned so any regime split can be done post hoc.  PRIVATE data stays under /tmp/nestquant/33-search/scale."""
import json
import os
import sys

import numpy as np
from scipy.signal import lfilter

sys.path.insert(0, "/home/coder/git/nestquant/threads/32-gbdt-sal")
import t32lib as T  # noqa: E402

G, NE, NBC = 16, 256, 512
SRC = "/tmp/nestquant/32-gbdt-sal"
BLK = f"{SRC}/private/sm120/blk"
OUT = "/tmp/nestquant/33-search/scale"
V2F = ["ema32", "ema128", "mem_cur_state", "tok_since_hit", "hits16", "sema32", "sema128", "sal16", "mps128"]
FIXED, FDEF = T.serve_sets()
LAYERS = T.LAYERS


def load(stream, L):
    """-> dict bc, bs (float64 [nb,NE]), bca, nans, segl, sg [(s,e)], names."""
    if stream in ("calib-fit", "glm52-heldout"):
        d = np.load(f"{SRC}/rows_bandall/{stream}/L{L}.npz")
        bc, bs = d["bcnt"].astype(np.float64), d["bsal"].astype(np.float64)
        z = np.load(f"{BLK}/{stream}/L{L}.npz")
        nb = bc.shape[0]
        bca = z["bcnta"][:nb].astype(np.float64)
        nans = z["nans"][:nb].astype(np.float64); segl = z["segl"][:nb].astype(np.int8)
        if len(nans) < nb:                                  # heldout: last partial window missing from blk (no think)
            pad = nb - len(nans)
            bca = np.vstack([bca, np.zeros((pad, NE))]); nans = np.r_[nans, np.zeros(pad)]; segl = np.r_[segl, np.zeros(pad, np.int8)]
        bst = list(range(0, nb, NBC)) + [nb]
        names = [f"{stream}#{i}" for i in range(len(bst) - 1)]
    else:
        z = np.load(f"{BLK}/{stream}/L{L}.npz")
        meta = json.load(open(f"{BLK}/{stream}/meta.json"))
        bc = z["bcnt"].astype(np.float64); bs = z["bsal"].astype(np.float64)
        bca = z["bcnta"].astype(np.float64); nans = z["nans"].astype(np.float64); segl = z["segl"].astype(np.int8)
        bst = meta["bstart"]; names = meta["chains"]
    sg = [(a, b) for a, b in zip(bst[:-1], bst[1:])]
    keep = [i for i, (a, b) in enumerate(sg) if b > a]
    return dict(bc=bc, bs=bs, bca=bca, nans=nans, segl=segl, sg=[sg[i] for i in keep], names=[names[i] for i in keep])


def ema_raw(M, h, sg):
    a = 0.5 ** (G / h)
    E = np.empty(M.shape, np.float64)
    for s, e in sg:
        E[s:e] = lfilter([1.0], [1.0, -a], M[s:e], axis=0)
    return E


def mem_state(bc, bca, nans, segl, sg, stuck=False):
    out = np.empty(bc.shape, np.float32)
    sa = 0.5 ** (1 / 2048)
    if stuck:
        a = sa ** G
        E = lfilter([1.0], [1.0, -a], bc, axis=0) if len(sg) == 1 else None
        for s, e in sg:
            Et = lfilter([1.0], [1.0, -a], bc[s:e], axis=0)
            wt = lfilter([1.0], [1.0, -a], np.full(e - s, float(G)))
            out[s:e] = Et / np.maximum(wt, 1e-6)[:, None]
        del E
        return out
    c32, ca = bc.astype(np.float32), bca.astype(np.float32)
    for s, e in sg:
        Et = np.zeros(NE, np.float32); Ea = np.zeros(NE, np.float32); wt = wa = 0.0
        for j in range(s, e):
            nt = G - nans[j]; dt = np.float32(sa ** nt); da = np.float32(sa ** nans[j])
            Et = Et * dt + (c32[j] - ca[j]); wt = wt * dt + nt; Ea = Ea * da + ca[j]; wa = wa * da + nans[j]
            out[j] = Et / max(wt, 1e-6) if segl[j] == 0 else Ea / max(wa, 1e-6)
    return out


def feats(D, stuck=False, prior=None):
    """-> F [nb, NE, 9] float32 in V2F order, e256 [nb,NE].  prior: optional dict of initial raw EMA states per chain
    (not used by default)."""
    bc, bs, sg = D["bc"], D["bs"], D["sg"]
    nb = bc.shape[0]
    ag = {h: 0.5 ** (G / h) for h in (32, 128, 256)}
    Ec = {h: ema_raw(bc, h, sg) for h in (32, 128, 256)}
    Es = {h: ema_raw(bs, h, sg) for h in (32, 128, 256)}
    norm = Es[256].sum(1) / np.maximum(Ec[256].sum(1), 1e-30)
    norm = np.where(norm > 0, norm, 1.0)[:, None]
    F = np.empty((nb, NE, 9), np.float32)
    for j, h in ((0, 32), (1, 128)):                         # serve float32 recurrence (bit parity with rows)
        a32 = np.float32(0.5 ** (G / h)); E = np.zeros(NE, np.float32); c32 = bc.astype(np.float32)
        fac = (1 - a32) / G
        for s, e in sg:
            E[:] = 0
            for t in range(s, e):
                E = E * a32 + c32[t]; F[t, :, j] = E * fac
    F[..., 2] = mem_state(bc, D["bca"], D["nans"], D["segl"], sg, stuck)
    k = np.arange(nb)
    for s, e in sg:
        last = np.maximum.accumulate(np.where(bc[s:e] > 0, k[s:e, None] - s, -10 ** 6), 0)
        F[s:e, :, 3] = np.minimum(G * (k[s:e, None] - s + 1 - last), 1e5)
    F[..., 4] = bc
    F[..., 5] = Es[32] * ((1 - ag[32]) / G) / norm
    F[..., 6] = Es[128] * ((1 - ag[128]) / G) / norm
    F[..., 7] = bs / norm
    h = Ec[128]
    F[..., 8] = np.where(h > 1e-3, Es[128] / np.maximum(h, 1e-30) / norm, 1.0)
    e256 = (Ec[256] * ((1 - ag[256]) / G)).astype(np.float32)
    return F, e256


def fut(M, n, sg):
    """sum of M over blocks k+1..k+n within chain (truncated) and valid mask (full horizon)."""
    out = np.zeros(M.shape); v = np.zeros(M.shape[0], bool)
    for s, e in sg:
        cs = np.vstack([np.zeros((1, M.shape[1])), np.cumsum(M[s:e], 0)])
        kk = np.arange(e - s)
        out[s:e] = cs[np.minimum(kk + 1 + n, e - s)] - cs[kk + 1]
        v[s:e] = kk + n < e - s
    return out, v


def cand_band(e256, fx, rlo=20, rhi=121):
    sc = np.where(fx[None], -np.inf, e256)
    return np.argsort(-sc, 1, kind="stable")[:, rlo:rhi]


def train_rows(D, F, e256, L, blocks=None, mL=None, target="sal", band="", sub=1):
    """v2 recipe rows: candidates EMA256 rank 20..120 of non-fixed (band '') or all non-fixed (band 'all');
    target next-64 salience / mL (or counts).  blocks: bool mask of eligible blocks."""
    fx = np.zeros(NE, bool); fx[FIXED[L]] = True
    Y, v = fut(D["bs"] if target == "sal" else D["bc"], 4, D["sg"])
    if blocks is not None:
        v &= blocks
    if sub > 1:
        v &= (np.arange(len(v)) % sub) == (L % sub)
    idx = np.flatnonzero(v)
    if band == "all":
        cand = np.broadcast_to(np.flatnonzero(~fx), (len(idx), int((~fx).sum())))
    else:
        cand = cand_band(e256[idx], fx)
    X = np.take_along_axis(F[idx], cand[..., None], 1).reshape(-1, F.shape[2])
    y = np.take_along_axis(Y[idx], cand, 1).ravel()
    if target == "sal":
        y = y / mL
    return X, y.astype(np.float32)


def predict_S(bst, F, L, cols=None, nthr=1, chunk=4096):
    fx = np.zeros(NE, bool); fx[FIXED[L]] = True
    nfx = np.flatnonzero(~fx)
    nb = F.shape[0]
    S = np.zeros((nb, NE), np.float32)
    for c0 in range(0, nb, chunk):
        X = F[c0:c0 + chunk][:, nfx]
        if cols is not None:
            X = X[..., cols]
        S[c0:c0 + chunk, nfx] = bst.predict(X.reshape(-1, X.shape[-1]), num_threads=nthr).reshape(-1, len(nfx))
    return S


def replay(S, L, sg, hm=0.5, cap=None, init=None, nf=51, lag=0):
    """t32lib.sim_layer (lag 0) semantics per chain; cap = decomp swap cap; init: optional [nchain, NE] bool initial
    floating set per chain (else floating_default)."""
    fx = np.zeros(NE, bool); fx[FIXED[L]] = True
    fd = np.zeros(NE, bool); fd[[e for e in FDEF[L] if e not in set(FIXED[L])][:nf]] = True
    nb = S.shape[0]
    serve = np.zeros((nb, NE), bool)
    for ci, (s, e) in enumerate(sg):
        want = (init[ci] & ~fx) if init is not None else fd.copy()
        for k in range(s, e):
            serve[k] = want
            if k - s < lag:
                continue
            Sk = S[k - lag]
            v = np.where(fx, -np.inf, Sk).astype(np.float32)
            r = want & ~fx
            if hm:
                v = np.where(r, v * np.float32(1 + hm), v)
            if np.where(fx, 0, np.maximum(Sk, 0)).sum() <= 0:
                nw = r
            else:
                o = np.argsort(-v, kind="stable")
                nw = np.zeros(NE, bool); nw[o[:nf]] = True
                if cap is not None:
                    new = np.nonzero(nw & ~want)[0]
                    if len(new) > cap:
                        keep_new = new[np.argsort(-v[new], kind="stable")[:cap]]
                        inc = np.nonzero(want)[0]
                        keep_inc = inc[np.argsort(-v[inc], kind="stable")[:nf - cap]]
                        nw = np.zeros(NE, bool); nw[keep_new] = True; nw[keep_inc] = True
            want = nw & ~fx
    return serve


def block_metrics(serve, D, L):
    """per block: covered all-slot salience, total salience, covered hits, total hits, churn (new floating vs the
    previous block, nan at k=0)."""
    fx = np.zeros(NE, bool); fx[FIXED[L]] = True
    hot = serve | fx
    bs, bc = D["bs"], D["bc"]
    ch = np.full(serve.shape[0], np.nan)
    ch[1:] = (serve[1:] & ~serve[:-1]).sum(1)
    return np.stack([(bs * hot).sum(1), bs.sum(1), (bc * hot).sum(1), bc.sum(1), ch], 1)


def oracle_S(D, n=4):
    return fut(D["bs"], n, D["sg"])[0].astype(np.float32)


def summarize(M, mask=None, cross_chain_churn=True):
    """M {L: [nb,5]} -> mean over layers of pooled sal-hot (and hits-hot), churn."""
    sal, cnt, ch = [], [], []
    for L, m in M.items():
        mk = np.ones(len(m), bool) if mask is None else (mask[L] if isinstance(mask, dict) else mask)
        a = m[mk]
        sal.append(a[:, 0].sum() / max(a[:, 1].sum(), 1e-30)); cnt.append(a[:, 2].sum() / max(a[:, 3].sum(), 1e-30))
        c = a[:, 4]; ch.append(np.nanmean(c) if np.isfinite(c).any() else np.nan)
    return dict(sal=float(np.mean(sal)) * 100, cnt=float(np.mean(cnt)) * 100, churn=float(np.nanmean(ch)))
