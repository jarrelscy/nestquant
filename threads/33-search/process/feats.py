"""process features for (corpus, L): feats(D, L) -> dict name -> [nb, NE] f32.  build: feats.py CORPUS -> $OUT/private/feat"""
import os, sys, time
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import plib as P, pfeat as Q
PF = ("hk_mu", "hk_lam", "hk_pred", "hmm_pred", "hmm_hot", "hmm_rise", "hmm_p0", "bo_cp", "bo_er", "bo_rate",
      "bo_rmap", "kf_lvl", "kf_slp", "kf_fc")
FAM = dict(hk=[f for f in PF if f.startswith("hk")], hmm=[f for f in PF if f.startswith("hmm")],
           bo=[f for f in PF if f.startswith("bo")], kf=[f for f in PF if f.startswith("kf")])
KF_QS = 1e-4
BO = dict(kappa=128.0, hazard=1 / 32, tau=16.0, R=128)


def feats(D, L, fams=("hk", "hmm", "bo", "kf")):
    p = np.load(f"{P.OUT}/params/L{L}.npz")
    C, segs = D["bcnt"], D["segs"]
    nb = C.shape[0]
    out = {}
    if "hk" in fams:
        out.update(Q.hawkes_feats(C, segs, dict(mu=p["hk_mu"], a=p["hk_a"])))
    if "hmm" in fams:
        out.update(Q.hmm_feats(C, segs, dict(lam=p["hmm_lam"], A=p["hmm_A"], pi=p["hmm_pi"])))
    if "bo" in fams:
        o = Q.bocpd_feats(C, segs, p["prior"], **BO)
        o.pop("evid")
        for k in ("bo_cp", "bo_er"):
            o[k] = np.broadcast_to(o[k][:, None], (nb, P.NE))
        out.update(o)
    if "kf" in fams:
        Y = np.log(D["F"]["sal16"].astype(np.float64) + 0.1)
        out.update(Q.kalman_feats(Y, segs, float(p["kf"][0]), KF_QS))   # MSE-best q_s is 0 (no slope); keep a slow slope as feature
    return {k: np.asarray(v, np.float32) for k, v in out.items()}


def job(a):
    corpus, L = a
    f = f"{P.OUT}/private/feat/{corpus}/L{L}.npz"
    if os.path.exists(f):
        return L, "exists"
    t = time.time()
    D = P.load(corpus, L)
    F = feats(D, L)
    np.savez(f, **{k: v.astype(np.float16) for k, v in F.items()})
    return L, round(time.time() - t)


if __name__ == "__main__":
    from multiprocessing import Pool
    corpus = sys.argv[1]
    os.makedirs(f"{P.OUT}/private/feat/{corpus}", exist_ok=True)
    with Pool(int(os.environ.get("NPROC", "16"))) as pool:
        for r in pool.imap_unordered(job, [(corpus, L) for L in P.T.LAYERS]):
            print(r, flush=True)
