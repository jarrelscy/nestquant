"""T33l ceiling: shared helpers (replay = T32 decomp.replay; scoring = all-slot sal-hot)."""
import os
import sys
import numpy as np
sys.path.insert(0, "/home/coder/git/nestquant/threads/32-gbdt-sal")
import t32lib as T  # noqa: E402

OUTC = "/tmp/nestquant/33-search/ceiling"
V2 = "/tmp/nestquant/32-gbdt-sal/models/v2_sal_tweedie1.5.txt"
NBC = T.CHAIN * T.SEQ // T.G
NF = 51
FIXED, FDEF = T.serve_sets()


def masks(L):
    fx = np.zeros(T.NE, bool); fx[FIXED[L]] = True
    fd = np.zeros(T.NE, bool); fd[[e for e in FDEF[L] if e not in set(FIXED[L])][:NF]] = True
    return fx, fd


def replay(S, fx, fd, hm=0.0, cap=None, nbc=NBC, dm=0.0):
    """T32 decomp.replay (r=1, lag=0): S[k] decided at end of block k serves block k+1."""
    nb = S.shape[0]
    serve = np.zeros((nb, T.NE), bool)
    for c0 in range(0, nb, nbc):
        want = fd.copy()
        for k in range(c0, min(c0 + nbc, nb)):
            serve[k] = want
            v = np.where(fx, -np.inf, S[k]).astype(np.float64)
            if hm:
                v = np.where(want, v * (1 + hm), v)
            if dm:                                   # additive margin relative to the 51st-best score (scale-free)
                fin = v[np.isfinite(v)]
                v51 = np.partition(fin, len(fin) - NF)[len(fin) - NF] if len(fin) >= NF else 0.0
                v = np.where(want, v + dm * max(v51, 1e-12), v)
            order = np.argsort(-v, kind="stable")
            nw = np.zeros(T.NE, bool); nw[order[:NF]] = True
            nw &= np.isfinite(v)
            if cap is not None:
                new = np.nonzero(nw & ~want)[0]
                if len(new) > cap:
                    keep_new = new[np.argsort(-v[new], kind="stable")[:cap]]
                    inc = np.nonzero(want)[0]
                    keep_inc = inc[np.argsort(-v[inc], kind="stable")[:NF - cap]]
                    nw = np.zeros(T.NE, bool); nw[keep_new] = True; nw[keep_inc] = True
            want = nw & ~fx
    return serve


def fut(M, n, nbc=NBC):
    """sum of M over blocks k+1 .. k+n within the chain (truncated at chain end)."""
    out = np.zeros(M.shape)
    for c0 in range(0, M.shape[0], nbc):
        s = M[c0:c0 + nbc]
        cs = np.vstack([np.zeros((1, M.shape[1])), np.cumsum(s, 0)])
        k = np.arange(s.shape[0])
        out[c0:c0 + nbc] = cs[np.minimum(k + 1 + n, s.shape[0])] - cs[np.minimum(k + 1, s.shape[0])]
    return out


def past(M, n, nbc=NBC):
    out = np.zeros(M.shape)
    for c0 in range(0, M.shape[0], nbc):
        s = M[c0:c0 + nbc]
        cs = np.vstack([np.zeros((1, M.shape[1])), np.cumsum(s, 0)])
        k = np.arange(s.shape[0])
        out[c0:c0 + nbc] = cs[k + 1] - cs[np.maximum(k + 1 - n, 0)]
    return out


def metric(sv, fx, Y):
    """all-slot sal-hot share of Y [nb,NE] (fixed always hot) + churn."""
    h = sv | fx
    return float((Y * h).sum() / Y.sum()), float((sv[1:] & ~sv[:-1]).sum(1).mean())


def v2_scores(corpus, L):
    f = f"{OUTC}/cache/{corpus}/L{L}.v2S.npy"
    if os.path.exists(f):
        return np.load(f)
    import lightgbm as lgb
    d = np.load(f"{T.OUT}/rows_bandall/{corpus}/L{L}.npz")
    b = lgb.Booster(model_file=V2)
    S = T.score_blocks(b.predict(T.feature_matrix(b.feature_name(), corpus, L, band="all", d=d), num_threads=1),
                       d["cand"], d["top"], d["e256"])
    os.makedirs(os.path.dirname(f), exist_ok=True)
    np.save(f, S)
    return S


def at_churn(pts, target=3.2):
    """pts [(sal, churn)] from a hysteresis grid -> sal linearly interpolated at churn=target."""
    p = sorted(pts, key=lambda x: x[1])
    c = [x[1] for x in p]; s = [x[0] for x in p]
    if target <= c[0]:
        return s[0], "extrap-lo"
    if target >= c[-1]:
        return s[-1], "extrap-hi"
    return float(np.interp(target, c, s)), "ok"
