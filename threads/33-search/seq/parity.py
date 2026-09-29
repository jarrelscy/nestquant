import os, sys, json
os.environ.setdefault("OMP_NUM_THREADS", "1")
from multiprocessing import Pool
import numpy as np
import common as C
corpus = sys.argv[1]
def job(L):
    bc, bs = C.load_blocks(corpus, L)
    S = C.v2_scores(corpus, L)
    return L, C.metrics(S, L, bc, bs, C.segs_of(corpus, bc.shape[0]))
if __name__ == "__main__":
    with Pool(12) as p:
        r = dict(p.map(job, C.LAYERS))
    print(corpus, "v2 sal-hot %.2f churn %.3f" % (100*np.mean([r[L]["sal"] for L in r]), np.mean([r[L]["churn"] for L in r])))
