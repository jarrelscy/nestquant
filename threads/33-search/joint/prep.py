"""prep.py STREAM [SUB] -> $OUT/feat/STREAM[_sSUB]/L{L}.npz : X [nb,256,K] fp16 (jlib.INPUTS), y16/y64/y256 f32
(next-k-token salience / m_L, nan beyond chain), keep [nb] block index.  PRIVATE."""
import os
import sys
import time
from multiprocessing import Pool
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import jlib as J

stream = sys.argv[1]; SUB = int(sys.argv[2]) if len(sys.argv) > 2 else 1
OD = f"{J.OUT}/feat/{stream}" + (f"_s{SUB}" if SUB > 1 else "")


def job(L):
    f = f"{OD}/L{L}.npz"
    if os.path.exists(f):
        return L, 0
    t0 = time.time()
    d = J.load(stream, L)
    F = J.features(d); fx, _ = J.masks(L)
    P = J.v2_pred(F, L, fx)
    X = J.net_inputs(F, P)
    bs = d["bsal"].astype(np.float64) / J.mL[L]
    ys = {f"y{k * 16}": J.future(bs, d["sg"], k) for k in (1, 4, 16)}
    keep = np.arange(X.shape[0])
    if SUB > 1:
        keep = keep[keep % SUB == 0]
    np.savez(f + ".part.npz", X=X[keep], keep=keep, **{k: v[keep] for k, v in ys.items()})
    os.replace(f + ".part.npz", f)
    return L, time.time() - t0


if __name__ == "__main__":
    os.makedirs(OD, exist_ok=True)
    with Pool(int(os.environ.get("NPROC", "16"))) as p:
        for L, t in p.imap_unordered(job, J.LAYERS):
            print(L, round(t), end=" | ", flush=True)
