"""parity: v2 on heldout via own path (expect 74.77 / 2.78)."""
import os as _o
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _o.environ[_v] = "1"
import sys
from multiprocessing import Pool
import numpy as np
import alib as A

corpus = sys.argv[1] if len(sys.argv) > 1 else "glm52-heldout"


def job(L):
    import lightgbm as lgb
    b = lgb.Booster(model_file=A.V2)
    d = A.load(corpus, L)
    F = A.base_feats(d, L, corpus=corpus)
    S = A.predict(b, F, A.FEATS9)
    sv, fx = A.sim(S, L, d["sg"])
    return A.metrics(sv, fx, d["bsal"].astype(np.float64), d["sg"])


if __name__ == "__main__":
    with Pool(16) as p:
        r = p.map(job, A.LAYERS)
    print(corpus, "v2 sal-hot %.2f churn %.3f churn_in %.3f" % (100 * np.mean([x["sal"] for x in r]),
          np.mean([x["churn"] for x in r]), np.mean([x["churn_in"] for x in r])))
