"""T33k alloc: shared helpers.  Predictor fixed at v2 (cached S from cache_scores.py); what changes is what the 77
level-4 slots per layer buy (fixed-set size k, per-projection partial upgrades)."""
import os, sys, json
import numpy as np
sys.path.insert(0, "/home/coder/git/nestquant/threads/32-gbdt-sal")
import t32lib as T  # noqa: E402

A = "/tmp/nestquant/33-search/alloc"
CACHE = f"{A}/cache"
NBC = T.CHAIN * T.SEQ // T.G       # 512 blocks per chain
fixed, fdef = T.serve_sets()
BANDS = {"L3-6": range(3, 7), "L7-40": range(7, 41), "L41-77": range(41, 78)}


def load(corpus, L):
    d = np.load(f"{CACHE}/{corpus}/L{L}.npz")
    return d["S"], d["bsal"].astype(np.float64), d["bcnt"].astype(np.float64)


def f26_ranked(L):
    """manifest fixed-26 ranked by calib-fit salience (= T32 fixed_size.py)."""
    cal = np.load(f"{CACHE}/calib-fit/L{L}.npz")["bsal"].sum(0)
    return sorted(fixed[L], key=lambda e: -cal[e])


def sim(S, fx, start, nf, hm, lag=0, nbc=NBC):
    """= t32lib.sim_layer, generalised (fx = list of fixed, start = ordered start list).  -> serve [nb,256] bool
    (floating only)."""
    return T.sim_layer(S, fx, start, nf=nf, hm=hm, nbc=nbc, lag=lag)


def churn(sv):
    return float((sv[1:] & ~sv[:-1]).sum(1).mean())


def churn_nochain(sv, nbc=NBC):
    """new floating experts per refresh, excluding chain-start resets."""
    new = (sv[1:] & ~sv[:-1]).sum(1)
    k = np.arange(1, sv.shape[0])
    return float(new[k % nbc != 0].mean())


def chains_of(nb, sg=None, nbc=NBC):
    return sg if sg is not None else [(c0, min(c0 + nbc, nb)) for c0 in range(0, nb, nbc)]


def sim_seg(S, fx, start, nf, hm, sg=None, init=None):
    """= t32lib.sim_layer lag 0 over explicit chain segments sg [(s,e)] (default fixed 512-block chains).
    init: optional [nchain, 256] bool initial floating sets."""
    nb = S.shape[0]
    fxm = np.zeros(256, bool); fxm[list(fx)] = True
    fd = np.zeros(256, bool); fd[[e for e in start if not fxm[e]][:nf]] = True
    serve = np.zeros((nb, 256), bool)
    f1 = np.float32(1 + hm)
    for ci, (s, e) in enumerate(chains_of(nb, sg)):
        want = (init[ci] & ~fxm) if init is not None else fd.copy()
        for k in range(s, e):
            serve[k] = want
            Sk = S[k]
            v = np.where(fxm, -np.inf, Sk).astype(np.float32)
            r = want & ~fxm
            v = np.where(r, v * f1, v)
            if np.where(fxm, 0, np.maximum(Sk, 0)).sum() <= 0:
                nw = r
            else:
                nw = np.zeros(256, bool); nw[np.argsort(-v, kind="stable")[:nf]] = True
            want = nw & ~fxm
    return serve


def churn_seg(sv, sg=None):
    """new floating experts per refresh, within chains only."""
    tot = n = 0
    for s, e in chains_of(sv.shape[0], sg):
        tot += (sv[s + 1:e] & ~sv[s:e - 1]).sum(); n += max(e - s - 1, 0)
    return float(tot / max(n, 1))
