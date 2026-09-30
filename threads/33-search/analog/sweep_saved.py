"""hot_eval-style (churn over all consecutive blocks, = T32 hot_eval) sal-hot/churn from saved heldout scores.
  sweep_saved.py DIR hm1,hm2,..   (NQ_LAYOUT=k0 for 77-floating)"""
import sys
from multiprocessing import Pool
import numpy as np
import alib as A
D, HMS = sys.argv[1], [float(x) for x in sys.argv[2].split(",")]


def job(L):
    d = A.load("glm52-heldout", L); S = np.load(f"{D}/L{L}.npy").astype(np.float32)
    out = []
    for hm in HMS:
        sv, fx = A.sim(S, L, d["sg"], hm=hm); out.append(A.metrics(sv, fx, d["bsal"].astype(float), d["sg"]))
    return out


if __name__ == "__main__":
    with Pool(8) as p:
        R = p.map(job, A.LAYERS)
    for i, hm in enumerate(HMS):
        print(f"{A.LAYOUT} {D.split('/')[-1]:32s} hm{hm:.2f} sal-hot {100 * np.mean([r[i]['sal'] for r in R]):6.2f} "
              f"churn {np.mean([r[i]['churn'] for r in R]):5.2f}", flush=True)
