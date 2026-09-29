"""v2-score-only decision arms; per-layer (sal, churn) curves on calib-val (chains %4==3) and heldout.
 -> $W/arms_v2.npz  (R[corpus] [nL, nArm, 2])"""
import sys, numpy as np
from multiprocessing import Pool
import dlib as D

ARMS = [("mult", h, 0) for h in (0, .1, .2, .3, .4, .5, .6, .8, 1., 1.5)]
ARMS += [("add", b, 0) for b in (0, .05, .1, .2, .3, .5, .8, 1.2, 2.)]
ARMS += [("rel51", b, 0) for b in (.05, .1, .2, .3, .5, .8)]            # R = S + b * S_(51st)
ARMS += [("cap", h, c) for c in (2, 3, 4, 5) for h in (0, .2)]
ARMS += [("log", b, 0) for b in (.05, .1, .2, .3, .4, .6)]               # rank log(S+eps): mult on log
def VR(S, kind, p):
    if kind == "mult": return S, S * (1 + p)
    if kind == "add": return S, S + p
    if kind == "cap": return S, S * (1 + p)
    if kind == "log":
        l = np.log(S + 0.05); return l, l + p
    if kind == "rel51":
        s51 = -np.sort(-S, 1)[:, 76:77]   # ~51st non-fixed incl fixed approx; recomputed below properly
        return S, S + p * s51

def job(args):
    corpus, L = args
    S = D.load_S("v2", corpus, L).astype(np.float64); bs, _ = D.load_blk(corpus, L); fx = D.fx_mask(L); fd = D.fd_mask(L)
    rows = D.val_rows(S.shape[0]) if corpus == "calib-fit" else None
    Snf = np.where(fx, -np.inf, S); s51 = -np.sort(-Snf, 1)[:, 50:51]
    out = []
    for kind, p, c in ARMS:
        if kind == "rel51": V, R = S, S + p * s51
        else: V, R = VR(S, kind, p)
        sv = D.replay(V, R, fx, fd, D.NBC, c)
        out.append(D.evaluate(sv, bs, fx, rows))
    return out

if __name__ == "__main__":
    res = {}
    with Pool(20) as pool:
        for corpus in ("calib-fit", "glm52-heldout"):
            res[corpus] = np.array(pool.map(job, [(corpus, L) for L in D.LAYERS]))
    np.savez(f"{D.W}/arms_v2.npz", **{k.replace('-', '_'): v for k, v in res.items()}, arms=np.array([f"{a[0]}:{a[1]}:{a[2]}" for a in ARMS]))
    for i, a in enumerate(ARMS):
        print(f"{a[0]:6s} {a[1]:5} cap{a[2]}  " + "  ".join(f"{c[:5]} {res[c][:, i, 0].mean()*100:.2f}/{res[c][:, i, 1].mean():.2f}" for c in res))
