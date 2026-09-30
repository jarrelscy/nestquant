"""analog features for calib-fit (train chains: leave-own-chain-out; val chains: full library) and glm52-heldout.
  build_feats.py VARIANT K D TAU(none|float) NLIBCH [PROJ] [HL=64,512]
-> /tmp/nestquant/33-search/analog/feat/VARIANT/{corpus}/L{L}.npz  (fp16 [nb, NE] each: an_f64 an_p64 an_f256 an_dist)"""
import os as _o
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _o.environ[_v] = "1"
import json
import os
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
BANK = os.environ.get("BANK", "calib")      # calib | self | both : bank used for NON-calib queries (heldout, sm120tf)
BSTRIDE = int(os.environ.get("BSTRIDE", "1"))   # self-bank: keep every BSTRIDE-th block (bank size cap)
CAUSAL = os.environ.get("CAUSAL", "0") == "1"
CORPORA = os.environ.get("CORPORA", "calib-fit,glm52-heldout").split(",")
OD = f"{A.OUT}/feat/{var}"


def feats_for(lib, Z, qcid=None):
    V = lib.query(Z, k=K, tau=TAU, qcid=qcid)
    f64, p64, f256 = V[:, :256] * 8, V[:, 256:512] * 8, V[:, 512:] * 8
    return f64, p64, f256


def self_bank(clib, d, Z, both):
    """leave-own-chain-out bank from the SAME corpus (serve analogue: a bank of other requests' traffic), optionally
    concatenated with the calib bank.  Same PCA as the calib bank."""
    sg = d["sg"]; bs = d["bsal"].astype(np.float64)
    P64 = N.share(N.ema_rate(bs, 64, sg)); F64, F256, ok = N.futures(bs, sg)
    cid = N.chain_id(sg, len(Z))
    ok = ok & (np.arange(len(Z)) % BSTRIDE == 0)
    K_ = clib.proj(Z[ok]); V_ = torch.from_numpy(np.concatenate([F64, P64, F256], 1)[ok].astype(np.float32))
    kc = torch.from_numpy(cid[ok].astype(np.int64))
    if both:
        K_ = torch.cat([K_, clib.K]); V_ = torch.cat([V_, clib.V]); kc = torch.cat([kc, torch.full((len(clib.K),), -9)])
    kn = (K_ ** 2).sum(1)
    Q = clib.proj(Z)
    out = np.empty((len(Z), 768), np.float32); dist = np.empty(len(Z), np.float32)
    for c0 in range(0, len(Z), 2048):
        q = Q[c0:c0 + 2048]
        d2 = (q ** 2).sum(1, keepdim=True) - 2 * q @ K_.T + kn[None]
        qc = torch.from_numpy(cid[c0:c0 + 2048].astype(np.int64))
        bad = qc[:, None] == kc[None]
        if CAUSAL:                                   # bank = strictly earlier chains only (+ calib when both)
            bad |= (kc[None] > qc[:, None])
        d2 = d2.masked_fill(bad, float("inf"))
        dv, ix = torch.topk(d2, K, 1, largest=False)
        out[c0:c0 + 2048] = V_[ix].mean(1).numpy(); dist[c0:c0 + 2048] = dv.clamp_min(0).mean(1).numpy()
    return out[:, :256] * 8, out[:, 256:512] * 8, out[:, 512:] * 8, dist


def job(L):
    torch.set_num_threads(1)
    t0 = time.time()
    lib, c = N.build_library(L, D=D, nch=NLIB, proj=PROJ, hl=HL)
    Q = lib.proj(c["Z"])
    # neighbour distance (mean d2 of the k nearest / library median NN d2) as match quality
    out = {}
    for corpus in CORPORA:
        if corpus == "calib-fit":
            Z = c["Z"]
            qc = c["wid"]                                # every calib query excludes own chain + doc-sharing windows
        else:
            d = A.load(corpus, L)
            Z = (N.states(d["bsal"].astype(np.float64), d["sg"], HL) if N.ML == [0] else
                 N.states_ml(corpus, L, d["sg"], HL, d["bsal"])); qc = None
        if corpus != "calib-fit" and BANK in ("self", "both"):
            f64, p64, f256, dist = self_bank(lib, d, Z, BANK == "both")
            os.makedirs(f"{OD}/{corpus}", exist_ok=True)
            np.savez(f"{OD}/{corpus}/L{L}.npz", an_f64=f64.astype(np.float16), an_p64=p64.astype(np.float16),
                     an_f256=f256.astype(np.float16), an_dist=dist)
            continue
        f64, p64, f256 = feats_for(lib, Z, qc)
        # match distance
        Qz = lib.proj(Z)
        dist = np.empty(len(Z), np.float32)
        for c0 in range(0, len(Z), 4096):
            q = Qz[c0:c0 + 4096]
            d2 = (q ** 2).sum(1, keepdim=True) - 2 * q @ lib.K.T + lib.kn[None]
            if qc is not None:
                qw = torch.from_numpy(qc[c0:c0 + 4096].astype(np.int64))
                d2 = d2.masked_fill(N.CONFLICT_T[qw][:, lib.cid], float("inf"))
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
