"""parity: v2 on glm52-heldout with jlib features + replay -> expect 74.77 / 2.78."""
import os
import sys
from multiprocessing import Pool
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import jlib as J


def job(L):
    d = J.load(sys.argv[1], L)
    F = J.features(d)
    fx, _ = J.masks(L)
    P = J.v2_pred(F, L, fx)
    return J.metric(P, d, L)


if __name__ == "__main__":
    with Pool(16) as p:
        r = np.array(p.map(job, J.LAYERS))
    print("sal-hot %.2f churn %.3f" % (100 * np.mean(r[:, 0] / r[:, 1]), np.mean(r[:, 2] / r[:, 3])))
