import os, sys, time
os.environ["OMP_NUM_THREADS"] = "2"
import numpy as np
import plib as P, pfeat as Q
L = int(sys.argv[1])
D = P.load("calib-fit", L)
segs = D["segs"]; tr = [sg for i, sg in enumerate(segs) if not Q.val_chain(i)]
fx = np.zeros(256, bool); fx[P.fixed[L]] = True
C = D["bcnt"]
t = time.time(); hk = Q.fit_hawkes(C, tr, fx); print("hawkes", hk["a"], hk["nll"], "mu mean", hk["mu"].mean(), f"{time.time()-t:.0f}s", flush=True)
t = time.time(); hm = Q.fit_hmm(C, tr[:8], fx, iters=15); print("hmm", hm["lam"], np.diag(hm["A"]), hm["pi"], f"{time.time()-t:.0f}s", flush=True)
prior = C[np.concatenate([np.arange(s, e) for s, e in tr])].sum(0); prior = prior / prior.sum()
for kap in (32, 128, 512):
    for hz in (1/32, 1/128, 1/512):
        t = time.time(); o = Q.bocpd_feats(C, tr[:3], prior, kap, hz); print("bocpd", kap, round(1/hz), round(o["evid"]), "cp", o["bo_cp"][:1536].mean().round(3), "Er", o["bo_er"][:1536].mean().round(1), f"{time.time()-t:.0f}s", flush=True)
