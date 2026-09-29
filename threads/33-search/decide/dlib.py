"""T33e decision-layer lib: fast replay of floating-set decision rules from cached per-block scores."""
import os, sys
sys.path.insert(0, "/home/coder/git/nestquant/threads/32-gbdt-sal")
import numpy as np
import numba as nb
import t32lib as T

W = "/tmp/nestquant/33-search/decide"
NBC = T.CHAIN * T.SEQ // T.G
NE, NF = 256, 51
FIXED, FDEF = T.serve_sets()
LAYERS = T.LAYERS


def fx_mask(L):
    m = np.zeros(NE, bool); m[FIXED[L]] = True; return m


def fd_mask(L):
    fx = set(FIXED[L]); m = np.zeros(NE, bool)
    m[[e for e in FDEF[L] if e not in fx][:NF]] = True
    return m


def load_blk(corpus, L):
    d = np.load(f"{W}/blk/{corpus}/L{L}.npz")
    return d["bsal"].astype(np.float64), d["bcnt"].astype(np.float64)


def load_S(name, corpus, L):
    return np.load(f"{W}/S/{name}/{corpus}/L{L}.npy")


@nb.njit(cache=True)
def replay(V, R, fx, fd, nbc, cap):
    """V [nb,NE] rank value if not resident, R [nb,NE] if resident (hysteresis applied by caller).
    serve[k] = set serving block k; decision at end of block k (sync, lag 0) serves k+1.  cap>0: max new per refresh.
    returns serve bool [nb,NE] (floating only)."""
    nbk, ne = V.shape
    serve = np.zeros((nbk, ne), np.bool_)
    want = fd.copy()
    v = np.empty(ne)
    for k in range(nbk):
        if k % nbc == 0:
            want = fd.copy()
        serve[k] = want
        for e in range(ne):
            if fx[e]:
                v[e] = -np.inf
            elif want[e]:
                v[e] = R[k, e]
            else:
                v[e] = V[k, e]
        o = np.argsort(-v, kind="mergesort")
        nw = np.zeros(ne, np.bool_)
        if cap <= 0:
            for i in range(NF):
                nw[o[i]] = True
        else:
            nnew = 0; n = 0
            for i in range(ne):
                e = o[i]
                if fx[e]:
                    continue
                if not want[e]:
                    if nnew >= cap:
                        continue
                    nnew += 1
                nw[e] = True; n += 1
                if n == NF:
                    break
        want = nw
    return serve


def evaluate(serve, bsal, fx, rows=None):
    """-> (sal-hot fraction, churn) ; churn = hot_eval def (mean new floating per transition, all transitions).
    rows: optional bool [nb] block mask (e.g. validation chains)."""
    s = serve.copy(); s[:, fx] = True
    new = (serve[1:] & ~serve[:-1]).sum(1)
    if rows is None:
        return (bsal * s).sum() / bsal.sum(), new.mean()
    return (bsal[rows] * s[rows]).sum() / bsal[rows].sum(), new[rows[1:]].mean()


def val_rows(nb, mod=4, r=3):
    """calib-fit validation chains: chain index % mod == r"""
    return (np.arange(nb) // NBC) % mod == r


def interp_at(sal, ch, target=3.2):
    """curve points (sal, churn) -> sal at churn target by linear interp on churn (sorted)."""
    o = np.argsort(ch); ch = np.asarray(ch)[o]; sal = np.asarray(sal)[o]
    return float(np.interp(target, ch, sal))
