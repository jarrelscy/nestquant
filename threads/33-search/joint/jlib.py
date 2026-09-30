"""T33i (joint): shared data/feature/eval library.  Vectorised causal per-block features over all 256 experts of a
layer (v2's 9 serve features + multi-half-life hit / salience EMAs + v2 log prediction), targets, and the sync
(lag 0) hysteresis replay + all-slot salience-hot metric.  PRIVATE outputs under /tmp/nestquant/33-search/joint."""
import json
import os
import sys

import numpy as np
from scipy.signal import lfilter

sys.path.insert(0, "/home/coder/git/nestquant/threads/32-gbdt-sal")
T32 = "/tmp/nestquant/32-gbdt-sal"
BLK = f"{T32}/private/sm120/blk"
OUT = "/tmp/nestquant/33-search/joint"
G, NE = 16, 256
LAYOUT = os.environ.get("LAYOUT", "k26")          # k26: 26 fixed + 51 floating (serve today) | k0: 77 floating
NF = 77 if LAYOUT == "k0" else 51
LAYERS = list(range(3, 78))
V2 = f"{T32}/models/v2_sal_tweedie1.5.txt"
mL = {int(k): v for k, v in json.load(open(V2 + ".meta.json"))["sal_norm_mL"].items()}
MANIFEST = ("/tmp/nestquant/32-gbdt-sal/k0_manifest.json" if LAYOUT == "k0" else
            "/tmp/nestquant/28-serve-release/out/serving/tp4/manifest.json")


def serve_sets():
    m = json.load(open(MANIFEST))
    fixed = {int(L): sorted(map(int, v)) for L, v in m["default_allocation"].items()}
    fdef = {int(L): [int(e) for e in v] for L, v in m["floating_default"].items()}
    return fixed, fdef


FIXED, FDEF = serve_sets()


def masks(L):
    fx = np.zeros(NE, bool); fx[FIXED[L]] = True
    fd = np.zeros(NE, bool); fd[[e for e in FDEF[L] if e not in set(FIXED[L])][:NF]] = True
    return fx, fd


def _blk_from_trace(stream, L):
    """calib-fit / glm52-heldout: block matrices from the T32 trace (t32lib semantics), cached (PRIVATE)."""
    f = f"{OUT}/blk/{stream}/L{L}.npz"
    if not os.path.exists(f):
        import t32lib as T
        ids, w, xn = T.load_layer(L, stream)
        tok = T.tokens(stream, ids.shape[0] // T.SEQ)
        seg = T.seg_of(tok)
        cnt, cnta, nans, sal, seg_last = T.block_mats(ids, w, xn, seg)
        os.makedirs(os.path.dirname(f), exist_ok=True)
        np.savez(f + ".part.npz", bcnt=cnt.astype(np.uint8), bcnta=cnta.astype(np.uint8), nans=nans.astype(np.uint8),
                 segl=seg_last.astype(np.uint8), bsal=sal.astype(np.float32))
        os.replace(f + ".part.npz", f)
    d = np.load(f)
    r = {k: d[k] for k in d.files}
    nb = r["bcnt"].shape[0]; nbc = 4 * 2048 // G
    r["sg"] = [(a, min(a + nbc, nb)) for a in range(0, nb, nbc)]
    return r


def load(stream, L):
    """-> dict bcnt, bcnta, nans, segl, bsal, sg list of (s, e) chain block ranges."""
    if stream in ("calib-fit", "glm52-heldout"):
        return _blk_from_trace(stream, L)
    d = np.load(f"{BLK}/{stream}/L{L}.npz")
    meta = json.load(open(f"{BLK}/{stream}/meta.json"))
    bs = meta["bstart"]
    sg = [(a, b) for a, b in zip(bs[:-1], bs[1:]) if b > a]
    r = {k: d[k] for k in ("bcnt", "bcnta", "nans", "segl", "bsal")}
    r["sg"] = sg
    return r


def ema(M, h, sg):
    a = 0.5 ** (G / h)
    E = np.empty(M.shape, np.float64)
    for s, e in sg:
        E[s:e] = lfilter([1.0], [1.0, -a], M[s:e], axis=0) * ((1 - a) / G)
    return E


HC = (8, 32, 64, 128, 256, 512, 2048)      # hit EMA half-lives (tokens)
HS = (8, 32, 64, 128, 256, 512, 2048)      # salience EMA half-lives
V2F = ("ema32", "ema128", "mem_cur_state", "tok_since_hit", "hits16", "sema32", "sema128", "sal16", "mps128")


def features(d):
    """-> dict name -> [nb, NE] float32 raw (v2 features exact serve definitions) + extras."""
    sg = d["sg"]
    bc = d["bcnt"].astype(np.float64); bs = d["bsal"].astype(np.float32).astype(np.float64)
    nb = bc.shape[0]
    F = {}
    Ec = {h: ema(bc, h, sg) for h in HC + ((256,) if 256 not in HC else ())}
    Es = {h: ema(bs, h, sg) for h in HS}
    nrm = Es[256].sum(1) / np.maximum(Ec[256].sum(1), 1e-30)
    nrm = np.where(nrm > 0, nrm, 1.0)[:, None]
    for h in HC:
        F[f"ema{h}"] = Ec[h]
    for h in (32, 128):                                 # serve arithmetic (float32 recursion) for the v2 inputs
        a = np.float32(0.5 ** (G / h))
        E = np.empty(bc.shape, np.float32)
        for s, e in sg:
            E[s:e] = lfilter(np.array([1.0], np.float32), np.array([1.0, -a], np.float32), bc[s:e].astype(np.float32), axis=0)
        F[f"ema{h}"] = E * ((1 - a) / G)
    for h in HS:
        F[f"sema{h}"] = Es[h] / nrm
    F["hits16"] = bc
    F["sal16"] = bs / nrm
    a128 = (1 - 0.5 ** (G / 128)) / G
    F["mps128"] = np.where(Ec[128] / a128 > 1e-3, Es[128] / np.maximum(Ec[128], 1e-30) / nrm, 1.0)
    a512 = (1 - 0.5 ** (G / 512)) / G
    F["mps512"] = np.where(Ec[512] / a512 > 1e-3, Es[512] / np.maximum(Ec[512], 1e-30) / nrm, 1.0)
    k = np.arange(nb)
    tsh = np.empty((nb, NE))
    for s, e in sg:
        last = np.maximum.accumulate(np.where(bc[s:e] > 0, k[s:e, None] - s, -10 ** 6), 0)
        tsh[s:e] = np.minimum(G * (k[s:e, None] - s + 1 - last), 1e5)
    F["tok_since_hit"] = tsh
    ca = d["bcnta"].astype(np.float32); c32 = bc.astype(np.float32)
    na, sl = d["nans"].astype(np.float64), d["segl"]
    sa = 0.5 ** (1 / 2048)
    out = np.empty((nb, NE), np.float32)
    for s, e in sg:
        Et = np.zeros(NE, np.float32); Ea = np.zeros(NE, np.float32); wt = wa = 0.0
        for j in range(s, e):
            nt = G - na[j]; dt = np.float32(sa ** nt); da = np.float32(sa ** na[j])
            Et = Et * dt + (c32[j] - ca[j]); wt = wt * dt + nt; Ea = Ea * da + ca[j]; wa = wa * da + na[j]
            out[j] = Et / max(wt, 1e-6) if sl[j] == 0 else Ea / max(wa, 1e-6)
    F["mem_cur_state"] = out
    # block position within chain (log) - cheap, serve-causal
    pos = np.empty(nb)
    for s, e in sg:
        pos[s:e] = np.arange(e - s)
    F["_pos"] = np.broadcast_to(np.log1p(pos)[:, None], (nb, NE))
    return {n: np.asarray(v, np.float32) for n, v in F.items()}


def v2_pred(F, L, fx):
    import lightgbm as lgb
    b = lgb.Booster(model_file=V2)
    nfx = np.arange(NE)                                 # all experts (k0 layout needs the fixed ones too)
    X = np.stack([F[n][:, nfx] for n in b.feature_name()], -1).reshape(-1, 9)
    p = b.predict(X, num_threads=int(os.environ.get("PT", "1"))).reshape(-1, len(nfx))
    P = np.zeros(F["ema32"].shape, np.float32); P[:, nfx] = p
    return P


def future(M, sg, k):
    """sum of blocks b+1..b+k within chain (nan beyond)."""
    Y = np.full(M.shape, np.nan, np.float32)
    for s, e in sg:
        cs = np.vstack([np.zeros((1, M.shape[1])), np.cumsum(M[s:e], 0)])
        kb = np.arange(max(e - s - k, 0))
        Y[s + kb] = cs[kb + 1 + k] - cs[kb + 1]
    return Y


# network input: log-compressed rates.  order fixed = INPUTS
RATE = [f"ema{h}" for h in HC] + [f"sema{h}" for h in HS] + ["hits16", "sal16"]
INPUTS = RATE + ["mem_cur_state", "tok_since_hit", "mps128", "mps512", "_pos", "v2"]


def net_inputs(F, P):
    cols = [np.log1p(np.maximum(F[n], 0) * (16.0 if n not in ("hits16", "sal16") else 1.0)) for n in RATE]
    cols += [np.log1p(np.maximum(F["mem_cur_state"], 0) * 16.0), np.log1p(F["tok_since_hit"]) / 5.0,
             np.log(np.clip(F["mps128"], 1e-3, None)), np.log(np.clip(F["mps512"], 1e-3, None)), F["_pos"] / 5.0,
             np.log(np.clip(P, 1e-4, None))]
    return np.stack(cols, -1).astype(np.float16)


# ------------------------------------------------------------------------------------------------ eval
def replay(S, fx, fd, sg, hm=0.5, ha=0.0):
    """sync lag-0 serve replay (= t32lib.sim_layer lag 0 / sm120.replay): S [nb, NE] >=0 scores -> serve [nb, NE]."""
    nb = S.shape[0]
    serve = np.zeros((nb, NE), bool)
    for s, e in sg:
        want = fd.copy()
        for k in range(s, e):
            serve[k] = want
            v = np.where(fx, -np.inf, S[k]).astype(np.float32)
            r = want & ~fx
            v = np.where(r, v * np.float32(1 + hm) + np.float32(ha), v)
            if np.where(fx, 0, np.maximum(S[k], 0)).sum() <= 0:
                nw = r
            else:
                nw = np.zeros(NE, bool); nw[np.argsort(-v, kind="stable")[:NF]] = True
            want = nw & ~fx
    return serve


def metric(S, d, L, hm=0.5, ha=0.0, sel=None):
    """-> (sal-hot numerator, denominator, churn sum, churn count)  pooled over chains (sel: chain indices)."""
    fx, fd = masks(L)
    sg = d["sg"] if sel is None else [d["sg"][i] for i in sel]
    sv = replay(S, fx, fd, sg, hm, ha)
    bs = d["bsal"].astype(np.float64)
    num = den = cs = cn = 0.0
    for s, e in sg:
        h = sv[s:e] | fx
        num += float((bs[s:e] * h).sum()); den += float(bs[s:e].sum())
        cs += float((sv[s + 1:e] & ~sv[s:e - 1]).sum()); cn += e - s - 1
        if s > 0 and sel is None:                       # hot_eval.py convention: churn over ALL block transitions
            cs += float((sv[s] & ~sv[s - 1]).sum()); cn += 1   # (incl. chain-start resets to floating_default)
    return num, den, cs, cn
