"""shared-mu Hawkes (no per-expert corpus prior) -> add hks_a, hks_mu to params/L.npz"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
from multiprocessing import Pool
import numpy as np
import plib as P, pfeat as Q, fit as FT

def job(L):
    f = f"{FT.OD}/L{L}.npz"
    z = dict(np.load(f))
    D = P.load("calib-fit", L)
    tr = [sg for i, sg in enumerate(D["segs"]) if not Q.val_chain(i)]
    fx = np.zeros(256, bool); fx[P.fixed[L]] = True
    hk = Q.fit_hawkes(D["bcnt"], tr, fx, shared_mu=True)
    z.update(hks_a=hk["a"], hks_mu=hk["mu"])
    np.savez(f, **z)
    return L, np.round(hk["a"], 3).tolist(), round(float(hk["mu"].max()), 4)

if __name__ == "__main__":
    with Pool(15) as p:
        for r in p.imap_unordered(job, P.T.LAYERS):
            print(r, flush=True)
