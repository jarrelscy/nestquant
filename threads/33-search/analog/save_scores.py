"""save heldout per-block scores (float16 [nblocks, 256]) for an arm: save_scores.py MODEL FEATVAR OUTDIR"""
import os as _o
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _o.environ[_v] = "1"
import os
import sys
import multiprocessing as mp
import numpy as np
import alib as A

model, fvar, od = sys.argv[1:4]


def job(L):
    import lightgbm as lgb
    b = lgb.Booster(model_file=model)
    F = A.base_feats(A.load("glm52-heldout", L), L, corpus="glm52-heldout")
    z = np.load(f"{A.OUT}/feat/{fvar}/glm52-heldout/L{L}.npz")
    for k in z.files:
        F[k] = z[k].astype(np.float32) if z[k].ndim == 2 else np.repeat(z[k][:, None], 256, 1).astype(np.float32)
    np.save(f"{od}/L{L}.npy", A.predict(b, F, b.feature_name()).astype(np.float16))
    return L


if __name__ == "__main__":
    os.makedirs(od, exist_ok=True)
    with mp.get_context("spawn").Pool(8) as p:
        p.map(job, A.LAYERS)
    print("saved", od)
