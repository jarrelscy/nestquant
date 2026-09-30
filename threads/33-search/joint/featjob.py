"""CPU feature job for on-the-fly streams (sm120tf): featjob.py STREAM OUTTMP -> OUTTMP/L.npz (X fp16, P f32)."""
import os, sys
from multiprocessing import Pool
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import jlib as J
stream, od = sys.argv[1], sys.argv[2]
LAY = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else J.LAYERS


def job(L):
    f = f"{od}/L{L}.npz"
    if os.path.exists(f) or os.path.exists(f + ".done"):
        return L
    d = J.load(stream, L); F = J.features(d); fx, _ = J.masks(L)
    P = J.v2_pred(F, L, fx)
    np.save(f"{J.OUT}/scores/v2_{stream}/L{L}.npy", P)
    X = J.net_inputs(F, P)
    np.savez(f + ".part.npz", X=X, P=P); os.replace(f + ".part.npz", f)
    return L


if __name__ == "__main__":
    os.makedirs(od, exist_ok=True); os.makedirs(f"{J.OUT}/scores/v2_{stream}", exist_ok=True)
    with Pool(int(os.environ.get("NPROC", "8"))) as p:
        for L in p.imap(job, LAY):
            print(L, end=" ", flush=True)
