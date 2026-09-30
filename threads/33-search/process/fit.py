"""fit per-layer process params on calib-fit TRAIN chains -> $OUT/params/L{L}.npz"""
import os, sys, time
os.environ["OMP_NUM_THREADS"] = "1"
from multiprocessing import Pool
import numpy as np
import plib as P, pfeat as Q
OD = f"{P.OUT}/params"; os.makedirs(OD, exist_ok=True)
C_OFF = 0.1


def kalman_grid(Y, tr, nf):
    best = None
    tgt = P.target(Y, tr) / 4.0
    ok = np.isfinite(tgt[:, 0])
    trr = np.zeros(len(ok), bool)
    for s, e in tr:
        trr[s:e] = True
    ok &= trr
    for ql in (0.003, 0.01, 0.03, 0.1, 0.3):
        for qs in (0.0, 1e-5, 1e-4, 1e-3, 1e-2):
            o = Q.kalman_feats(Y[:, nf], tr, ql, qs)
            for h in (0.0, 2.5):
                pr = o["kf_lvl"] + h * o["kf_slp"]
                mse = float(((pr[ok] - tgt[ok][:, nf]) ** 2).mean())
                if h == 2.5 or qs == 0.0:
                    if best is None or mse < best[0]:
                        best = (mse, ql, qs, h)
    return best


def job(L):
    f = f"{OD}/L{L}.npz"
    if os.path.exists(f):
        return L, "exists"
    t = time.time()
    D = P.load("calib-fit", L)
    segs = D["segs"]; tr = [sg for i, sg in enumerate(segs) if not Q.val_chain(i)]
    fx = np.zeros(256, bool); fx[P.fixed[L]] = True
    C = D["bcnt"]
    hk = Q.fit_hawkes(C, tr, fx)
    hm = Q.fit_hmm(C, tr, fx, iters=20)
    rows = np.concatenate([np.arange(s, e) for s, e in tr])
    prior = C[rows].sum(0); prior = prior / prior.sum()
    Y = np.log(D["F"]["sal16"].astype(np.float64) + C_OFF)
    kb = kalman_grid(Y, tr, ~fx)
    # also slope-free best for reference is inside grid; store best
    np.savez(f, hk_mu=hk["mu"], hk_a=hk["a"], hmm_lam=hm["lam"], hmm_A=hm["A"], hmm_pi=hm["pi"], prior=prior,
             kf=np.array(kb[1:3]), kf_mse=kb[0], kf_h=kb[3])
    return L, dict(hk_a=np.round(hk["a"], 3).tolist(), hmm_lam=np.round(hm["lam"], 3).tolist(), kf=kb, s=round(time.time() - t))


if __name__ == "__main__":
    with Pool(int(os.environ.get("NPROC", "16"))) as p:
        for r in p.imap_unordered(job, P.T.LAYERS):
            print(r, flush=True)
