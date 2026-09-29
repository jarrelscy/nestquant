"""analog features for calib-fit (train chains: leave-own-chain-out; val chains: full library) and glm52-heldout.
  build_feats.py VARIANT K D TAU(none|float) NLIBCH [PROJ] [HL=64,512]
-> /tmp/nestquant/33-search/analog/feat/VARIANT/{corpus}/L{L}.npz  (fp16 [nb, NE] each: an_f64 an_p64 an_f256 an_dist)"""
import os as _o
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _o.environ[_v] = "1"
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import torch

import alib as A
import analog as N

var, K, D = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
TAU = None if sys.argv[4] == "none" else float(sys.argv[4])
NLIB = int(sys.argv[5])
PROJ = sys.argv[6] if len(sys.argv) > 6 else "pca"
HL = tuple(int(x) for x in sys.argv[7].split(",")) if len(sys.argv) > 7 else (64, 512)
OD = f"{A.OUT}/feat/{var}"


def feats_for(lib, Z, qcid=None):
    V = lib.query(Z, k=K, tau=TAU, qcid=qcid)
    f64, p64, f256 = V[:, :256] * 8, V[:, 256:512] * 8, V[:, 512:] * 8
    return f64, p64, f256


def job(L):
    torch.set_num_threads(1)
    t0 = time.time()
    lib, c = N.build_library(L, D=D, nch=NLIB, proj=PROJ, hl=HL)
    Q = lib.proj(c["Z"])
    # neighbour distance (mean d2 of the k nearest / library median NN d2) as match quality
    out = {}
    for corpus in ("calib-fit", "glm52-heldout"):
        if corpus == "calib-fit":
            Z, cid = c["Z"], c["cid"]
            qc = np.where(cid < NLIB, cid, -1)          # train-chain queries exclude their own chain
        else:
            d = A.load(corpus, L)
            Z = N.states(d["bsal"].astype(np.float64), d["sg"], HL); qc = None
        f64, p64, f256 = feats_for(lib, Z, qc)
        # match distance
        Qz = lib.proj(Z)
        dist = np.empty(len(Z), np.float32)
        for c0 in range(0, len(Z), 4096):
            q = Qz[c0:c0 + 4096]
            d2 = (q ** 2).sum(1, keepdim=True) - 2 * q @ lib.K.T + lib.kn[None]
            if qc is not None:
                m = torch.from_numpy(qc[c0:c0 + 4096].astype(np.int64))[:, None] == lib.cid[None]
                d2 = d2.masked_fill(m, float("inf"))
            dist[c0:c0 + 4096] = torch.topk(d2, K, 1, largest=False)[0].clamp_min(0).mean(1).numpy()
        os.makedirs(f"{OD}/{corpus}", exist_ok=True)
        np.savez(f"{OD}/{corpus}/L{L}.npz", an_f64=f64.astype(np.float16), an_p64=p64.astype(np.float16),
                 an_f256=f256.astype(np.float16), an_dist=dist)
    return L, time.time() - t0


if __name__ == "__main__":

    os.makedirs(OD, exist_ok=True)
    json.dump(dict(K=K, D=D, tau=TAU, nlib=NLIB, proj=PROJ, hl=HL), open(f"{OD}/meta.json", "w"))
    with Pool(int(os.environ.get("NPROC", "16"))) as p:
        for L, s in p.imap_unordered(job, A.LAYERS):
            pass
    print("done", var, s)
