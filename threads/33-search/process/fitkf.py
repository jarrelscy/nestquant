"""re-fit Kalman grid only (fixes uninitialised-row bug) -> overwrite kf in params/L.npz"""
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
    Y = np.log(D["F"]["sal16"].astype(np.float64) + FT.C_OFF)
    kb = FT.kalman_grid(Y, tr, ~fx)
    z.update(kf=np.array(kb[1:3]), kf_mse=kb[0], kf_h=kb[3])
    np.savez(f, **z)
    return L, kb

if __name__ == "__main__":
    with Pool(15) as p:
        for r in p.imap_unordered(job, P.T.LAYERS):
            print(r, flush=True)
