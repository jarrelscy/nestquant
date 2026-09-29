#!/usr/bin/env python3
"""T33e: cache v2 per-block score matrices S [nb,256] (float32) per layer + per-block sal/cnt for a corpus.
  score_v2.py CORPUS -> /tmp/nestquant/33-search/decide/S/v2/CORPUS/L{L}.npy, blk/CORPUS/L{L}.npz"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, "/home/coder/git/nestquant/threads/32-gbdt-sal")
from multiprocessing import Pool
import numpy as np
import t32lib as T
corpus = sys.argv[1]
MODEL = "/tmp/nestquant/32-gbdt-sal/models/v2_sal_tweedie1.5.txt"
W = "/tmp/nestquant/33-search/decide"


def job(L):
    import lightgbm as lgb
    d = np.load(f"{T.OUT}/rows_bandall/{corpus}/L{L}.npz")
    b = lgb.Booster(model_file=MODEL)
    S = T.score_blocks(b.predict(T.feature_matrix(b.feature_name(), corpus, L, band="all", d=d), num_threads=1),
                       d["cand"], d["top"], d["e256"])
    os.makedirs(f"{W}/S/v2/{corpus}", exist_ok=True); os.makedirs(f"{W}/blk/{corpus}", exist_ok=True)
    np.save(f"{W}/S/v2/{corpus}/L{L}.npy", S)
    np.savez(f"{W}/blk/{corpus}/L{L}.npz", bsal=d["bsal"], bcnt=d["bcnt"])
    return L


if __name__ == "__main__":
    with Pool(16) as p:
        print(len(p.map(job, T.LAYERS)))
