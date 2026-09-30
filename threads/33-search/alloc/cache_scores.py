#!/usr/bin/env python3
"""T33k alloc: cache v2 per-block score matrices S [nb,256] f32 (GBDTPredictor._score output, band all) + bsal/bcnt per
(corpus, layer) -> /tmp/nestquant/33-search/alloc/cache/{corpus}/L{L}.npz.  Predictor fixed at v2.
  cache_scores.py CORPUS [MODEL]"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, "/home/coder/git/nestquant/threads/32-gbdt-sal")
import numpy as np
from multiprocessing import Pool
import t32lib as T

corpus = sys.argv[1]
MODEL = sys.argv[2] if len(sys.argv) > 2 else "/tmp/nestquant/32-gbdt-sal/models/v2_sal_tweedie1.5.txt"
ROWS = os.environ.get("ROWS", f"{T.OUT}/rows_bandall")
OUTD = f"/tmp/nestquant/33-search/alloc/cache/{corpus}"


def job(L):
    import lightgbm as lgb
    f = f"{OUTD}/L{L}.npz"
    if os.path.exists(f):
        return L
    d = np.load(f"{ROWS}/{corpus}/L{L}.npz")
    b = lgb.Booster(model_file=MODEL)
    S = T.score_blocks(b.predict(T.feature_matrix(b.feature_name(), corpus, L, band="all", d=d), num_threads=1),
                       d["cand"], d["top"], d["e256"])
    np.savez(f + ".tmp.npz", S=S, bsal=d["bsal"].astype(np.float32), bcnt=d["bcnt"].astype(np.uint8))
    os.replace(f + ".tmp.npz", f)
    return L


if __name__ == "__main__":
    os.makedirs(OUTD, exist_ok=True)
    with Pool(int(os.environ.get("NPROC", "16"))) as p:
        for L in p.imap_unordered(job, T.LAYERS):
            pass
    print("done", corpus)
