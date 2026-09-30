"""uncertainty + value-of-swap arms.  quantile: V = (1-a) h64 + a q ; VoS: A = short blend, R = A + lam * r256
(resident protection by predicted long-horizon value) ; std: V = h64 + k * (q90 - h64) (upper-bound ranking)."""
import sys, numpy as np
from multiprocessing import Pool
import dlib as D
HMS = (0.2, 0.35, 0.5, 0.7, 1.0)
ARMS = [("v2", 0, h) for h in HMS]
ARMS += [(f"q90mix", a, h) for a in (0.25, 0.5, 1.0) for h in HMS]
ARMS += [(f"q70mix", a, h) for a in (0.25, 0.5, 1.0) for h in HMS]
ARMS += [(f"ub", k, h) for k in (-0.5, 0.25, 0.5) for h in HMS]
ARMS += [(f"vos", lam, 0) for lam in (0.5, 1, 2, 3, 4, 6, 8, 12)]
ARMS += [(f"vos16", lam, 0) for lam in (0.5, 1, 2, 3, 4, 6, 8, 12)]

def load(corpus, name, L):
    S = D.load_S(name, corpus, L).astype(np.float64)
    return S[D.val_rows(S.shape[0])] if corpus == "calib-fit" else S

def job(args):
    corpus, L = args
    bs, _ = D.load_blk(corpus, L)
    if corpus == "calib-fit": bs = bs[D.val_rows(bs.shape[0])]
    fx = D.fx_mask(L); fd = D.fd_mask(L)
    H = {n: load(corpus, n, L) for n in ("v2", "h16", "h64", "h256", "q64_90", "q64_70")}
    out = []
    for kind, p, hm in ARMS:
        if kind == "v2": V = H["v2"]
        elif kind == "q90mix": V = (1 - p) * H["h64"] + p * H["q64_90"]
        elif kind == "q70mix": V = (1 - p) * H["h64"] + p * H["q64_70"]
        elif kind == "ub": V = H["h64"] + p * (H["q64_90"] - H["h64"])
        if kind.startswith("vos"):
            A = H["h64"] / 4 if kind == "vos" else 0.5 * H["h16"] + 0.5 * H["h64"] / 4
            sv = D.replay(A, A + p * 0.1 * H["h256"] / 16, fx, fd, D.NBC, 0)
        else:
            sv = D.replay(V, V * (1 + hm), fx, fd, D.NBC, 0)
        out.append(D.evaluate(sv, bs, fx))
    return out

if __name__ == "__main__":
    res = {}
    with Pool(20) as pool:
        for corpus in ("calib-fit", "glm52-heldout"):
            res[corpus.replace("-", "_")] = np.array(pool.map(job, [(corpus, L) for L in D.LAYERS]))
    names = np.array([f"{k}:{p}:{h}" for k, p, h in ARMS])
    np.savez(f"{D.W}/arms2.npz", **res, arms=names)
    groups = {}
    for i, (k, p, h) in enumerate(ARMS):
        groups.setdefault(f"{k}:{p}" if not k.startswith("vos") else k, []).append(i)
    for g, idx in groups.items():
        c = res["calib_fit"][:, idx].mean(0); h = res["glm52_heldout"][:, idx].mean(0)
        print(f"{g:14s} calib@3.2 {D.interp_at(c[:,0], c[:,1])*100:.2f}  held@3.2 {D.interp_at(h[:,0], h[:,1])*100:.2f}  "
              f"held curve " + " ".join(f"{s*100:.2f}/{x:.2f}" for s, x in h))
