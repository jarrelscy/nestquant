"""predict_heads.py CORPUS HEAD [HEAD...] -> $W/S/HEAD/CORPUS/L{L}.npy  (per-block [nb,256] scores, expert-indexed)"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from multiprocessing import Pool
import dlib as D
T = D.T
corpus, heads = sys.argv[1], sys.argv[2:]

def job(L):
    import lightgbm as lgb
    d = np.load(f"{T.OUT}/rows_bandall/{corpus}/L{L}.npz")
    X = None
    for h in heads:
        b = lgb.Booster(model_file=f"{D.W}/models/{h}.txt")
        if X is None: X = T.feature_matrix(b.feature_name(), corpus, L, band="all", d=d)
        S = T.score_blocks(b.predict(X, num_threads=1), d["cand"], d["top"], d["e256"])
        os.makedirs(f"{D.W}/S/{h}/{corpus}", exist_ok=True)
        np.save(f"{D.W}/S/{h}/{corpus}/L{L}.npy", S)
    return L

if __name__ == "__main__":
    with Pool(12) as p: print(len(p.map(job, D.LAYERS)))
