#!/usr/bin/env python3
"""rows_last: CAUSAL persistence proxy of the MTP oracle: true routing of the LAST k tokens of block b
(16(b+1)-k .. 16(b+1)-1), k=1,2,4: cnt and salience / norm_L (same normalisation as build_x mtp*).  PRIVATE."""
import os
import sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multiprocessing import Pool  # noqa: E402
import numpy as np  # noqa: E402
import dlib as D  # noqa: E402
import t32lib as T  # noqa: E402
sys.path.insert(0, "/home/coder/git/nestquant/threads/32-gbdt-sal")
from build_x import norm_of  # noqa: E402

corpus = sys.argv[1]
out = f"{D.PRIV}/rows_last/{corpus}"
NAMES = ["last1_cnt", "last1_sal", "last2_cnt", "last2_sal", "last4_cnt", "last4_sal"]


def job(L):
    d = D.load_rows(corpus, L)
    bc, bs, cand = d["bcnt"].astype(np.float64), d["bsal"].astype(np.float64), d["cand"].astype(np.int64)
    nb = bc.shape[0]
    nrm = norm_of(bc, bs)
    ids, w, xn = T.load_layer(L, corpus)
    v = w.astype(np.float64) ** 2 * xn.astype(np.float64)[:, None]
    F = []
    for k in (1, 2, 4):
        mc, ms = np.zeros((nb, 256)), np.zeros((nb, 256))
        for j in range(1, k + 1):
            t = (np.arange(nb) + 1) * T.G - j
            b = np.arange(nb)
            np.add.at(mc, (np.repeat(b, 8), ids[t].astype(np.int64).ravel()), 1.0)
            np.add.at(ms, (np.repeat(b, 8), ids[t].astype(np.int64).ravel()), v[t].ravel())
        F += [mc, ms / nrm]
    Fa = np.stack([np.take_along_axis(f, cand, 1) for f in F], -1).astype(np.float32)
    os.makedirs(out, exist_ok=True)
    np.savez(f"{out}/L{L}.npz", F=Fa)
    return L


if __name__ == "__main__":
    with Pool(16) as p:
        print(len(p.map(job, T.LAYERS)))
