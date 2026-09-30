"""CPU: sm120tf training rows for decode-fold training, from tmp_tfX (X, P) + targets.
  tfprep.py SUB -> feat/sm120tf_s{SUB}/L.npz : X fp16 [n,256,22], P f32, y64 f16, task int8 (J.load sg index), blk"""
import os, sys
from multiprocessing import Pool
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import jlib as J
SUB = int(sys.argv[1]); OD = f"{J.OUT}/feat/sm120tf_s{SUB}"; TD = f"{J.OUT}/tmp_tfX"


def job(L):
    f = f"{OD}/L{L}.npz"
    if os.path.exists(f):
        return L
    d = J.load("sm120tf", L)
    y = J.future(d["bsal"].astype(np.float64) / J.mL[L], d["sg"], 4)
    task = np.full(y.shape[0], -1, np.int8)
    for t, (s, e) in enumerate(d["sg"]):
        task[s:e] = t
    k = np.arange(y.shape[0]); keep = k[(k % SUB == 0) & np.isfinite(y).all(1) & (task >= 0)]
    z = np.load(f"{TD}/L{L}.npz")
    np.savez(f + ".part.npz", X=z["X"][keep], P=z["P"][keep], y64=y[keep].astype(np.float16), task=task[keep], blk=keep)
    os.replace(f + ".part.npz", f)
    return L


if __name__ == "__main__":
    os.makedirs(OD, exist_ok=True)
    with Pool(int(os.environ.get("NPROC", "6"))) as p:
        for L in p.imap_unordered(job, J.LAYERS):
            print(L, end=" ", flush=True)
    print("done")
