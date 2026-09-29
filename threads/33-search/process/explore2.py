import os, sys, time
os.environ["OMP_NUM_THREADS"] = "2"
import numpy as np
import plib as P, pfeat as Q
L = int(sys.argv[1])
D = P.load("calib-fit", L)
segs = D["segs"]; tr = [sg for i, sg in enumerate(segs) if not Q.val_chain(i)]
C = D["bcnt"]
prior = C[np.concatenate([np.arange(s, e) for s, e in tr])].sum(0); prior = prior / prior.sum()
for tau in (4, 16, 64):
  for kap in (128, 512):
    for hz in (1/32, 1/256):
        o = Q.bocpd_feats(C, tr[:3], prior, kap, hz, tau=tau); print("bocpd tau", tau, kap, round(1/hz), round(o["evid"]*tau), "cp", o["bo_cp"][:1536].mean().round(3), "Er", o["bo_er"][:1536].mean().round(1), flush=True)
