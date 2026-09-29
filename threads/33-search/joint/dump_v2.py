"""exact v2 scores per layer -> $OUT/scores/v2_STREAM/L.npy"""
import os, sys
from multiprocessing import Pool
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import jlib as J
s = sys.argv[1]; od = f"{J.OUT}/scores/v2_{s}"


def job(L):
    d = J.load(s, L); F = J.features(d); fx, _ = J.masks(L)
    np.save(f"{od}/L{L}.npy", J.v2_pred(F, L, fx))


if __name__ == "__main__":
    os.makedirs(od, exist_ok=True)
    with Pool(16) as p:
        p.map(job, J.LAYERS)
