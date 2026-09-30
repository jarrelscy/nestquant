"""training rows from calib-fit: train chains (i%5!=4) / val chains, every SUB-th valid block, non-fixed experts.
-> $OUT/private/rows/L{L}.npz  Xtr [n, NF] f32, ytr, Xva, yva  (columns = COLS)"""
import os, sys
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import plib as P, pfeat as Q, feats as FE
COLS = P.V2F + list(FE.PF) + list(FE.PF2)
SUB = 4
RD = f"{P.OUT}/private/rows" + ("0" if P.LAYOUT == "k0" else "")


def job(L):
    f = f"{RD}/L{L}.npz"
    if os.path.exists(f):
        return L
    D = P.load("calib-fit", L)
    pf = np.load(f"{P.OUT}/private/feat/calib-fit/L{L}.npz")
    src = dict(D["F"]); src.update({k: pf[k] for k in pf.files})
    pf2 = np.load(f"{P.OUT}/private/feat2/calib-fit/L{L}.npz"); src.update({k: pf2[k] for k in pf2.files})
    Y = P.target(D["bsal"], D["segs"]) / D["mL"]
    nf = np.ones(256, bool); nf[P.fixed[L]] = False
    out = {}
    for tag, isval in (("tr", False), ("va", True)):
        b = np.concatenate([np.arange(s, e) for i, (s, e) in enumerate(D["segs"]) if Q.val_chain(i) == isval])
        b = b[(b % SUB == L % SUB) & np.isfinite(Y[b, 0])]
        out["X" + tag] = np.stack([src[c][b][:, nf] for c in COLS], -1).reshape(-1, len(COLS)).astype(np.float32)
        out["y" + tag] = Y[b][:, nf].ravel().astype(np.float32)
    np.savez(f, **out)
    return L


if __name__ == "__main__":
    from multiprocessing import Pool
    os.makedirs(RD, exist_ok=True)
    with Pool(int(os.environ.get("NPROC", "16"))) as p:
        print(sorted(p.map(job, P.T.LAYERS))[-1])
