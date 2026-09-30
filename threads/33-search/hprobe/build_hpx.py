#!/usr/bin/env python3
"""hpx rows (band all, = T32 rows_bandall + rows_v2_bandall arithmetic) + v2 scores, from private/trace_hpx.
  -> private/rows_hpx/L.npz (X [nb,256,9] = FEATS5+FEATS_V2, cand, top, e256, ysal, valid, bcnt, bsal), private/v2S_hpx/L.npy"""
import os
import sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
from multiprocessing import Pool  # noqa: E402
import numpy as np  # noqa: E402
import heval as H  # noqa: E402
import t32lib as T  # noqa: E402

P = f"{H.HP}/private"
T.CORP = f"{P}/corpora"
T.load_layer.__defaults__ = (f"{P}/trace_hpx",)
OUT = f"{P}/rows_hpx"


def job(L):
    import lightgbm as lgb
    tmp = f"{OUT}/tmp"
    T.build_layer(L, "hpx", H.FIXED, tmp, 0, 256)
    d = dict(np.load(f"{tmp}/L{L}.npz"))
    X2 = T.v2_features(d["bcnt"], d["bsal"], d["cand"])
    X = np.concatenate([d["X"], X2], -1)
    b = lgb.Booster(model_file=H.V2)
    S = T.score_blocks(b.predict(X.reshape(-1, X.shape[-1]), num_threads=1), d["cand"], d["top"], d["e256"])
    np.save(f"{P}/v2S_hpx/L{L}.npy", S)
    keep = {k: d[k] for k in ("cand", "top", "e256", "ysal", "valid", "bcnt", "bsal")}
    np.savez(f"{OUT}/L{L}.npz", X=X, **keep)
    os.remove(f"{tmp}/L{L}.npz")
    return L, H.eval_S(S, L, d["bsal"].astype(np.float64), d["bcnt"].astype(np.float64))


if __name__ == "__main__":
    os.makedirs(f"{OUT}/tmp", exist_ok=True); os.makedirs(f"{P}/v2S_hpx", exist_ok=True)
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 12) as p:
        res = dict(p.imap_unordered(job, T.LAYERS))
    print("v2 on hpx", H.summarise(res))
