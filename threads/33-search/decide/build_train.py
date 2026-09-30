"""training matrix for multi-horizon heads: calib-fit TRAIN chains (chain % 4 != 3), every STRIDE-th block, all 256
candidates (band all), v2's 9 features; targets (normalised by m_L): sal16 (next block), sal64, sal128, sal256, cnt64.
Also a validation subsample from val chains (stride 8). -> $W/train_s{STRIDE}.npz"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from multiprocessing import Pool
import dlib as D
T = D.T
STRIDE = int(sys.argv[1]) if len(sys.argv) > 1 else 4
FE = list(T.FEATS5) + list(T.FEATS_V2)

def futn(M, n):
    """sum of blocks b+1..b+n within chain; NaN where the horizon leaves the chain (= rows_lh semantics)"""
    out = np.full(M.shape, np.nan)
    for c0 in range(0, M.shape[0], D.NBC):
        s = M[c0:c0 + D.NBC]; cs = np.vstack([np.zeros((1, M.shape[1])), np.cumsum(s, 0)])
        k = np.arange(s.shape[0] - n); out[c0 + k] = cs[k + 1 + n] - cs[k + 1]
    return out


def job(L):
    c = "calib-fit"
    d = np.load(f"{T.OUT}/rows_bandall/{c}/L{L}.npz")
    X = T.feature_matrix(FE, c, L, band="all", d=d).reshape(-1, 256, len(FE))
    m = float(d["slot_sal_sum"] / d["slots"])
    bs = d["bsal"].astype(np.float64); nb = bs.shape[0]
    cand = d["cand"].astype(np.int64)
    y16 = np.full((nb, 256), np.nan); nxt = np.arange(nb) + 1
    ok = (nxt % D.NBC) != 0
    y16[ok] = bs[nxt[ok]]
    y16 = np.take_along_axis(y16, cand, 1)
    f128 = np.take_along_axis(futn(bs, 8), cand, 1); f256 = np.take_along_axis(futn(bs, 16), cand, 1)
    Y = np.stack([y16, d["ysal"], f128, f256], -1) / m
    Y = np.concatenate([Y, d["ycnt"][..., None]], -1).astype(np.float32)
    ch = np.arange(nb) // D.NBC; b = np.arange(nb)
    fin = np.isfinite(Y[:, 0, :4]).all(1)
    tr = (ch % 4 != 3) & (b % STRIDE == 0) & fin
    va = (ch % 4 == 3) & (b % 8 == 1) & fin
    return (X[tr].reshape(-1, len(FE)), Y[tr].reshape(-1, 5), np.full(tr.sum() * 256, L, np.int8),
            X[va].reshape(-1, len(FE)), Y[va].reshape(-1, 5), np.full(va.sum() * 256, L, np.int8))

if __name__ == "__main__":
    with Pool(12) as p: R = p.map(job, D.LAYERS)
    out = {k: np.concatenate([r[i] for r in R]) for i, k in enumerate(["Xt", "Yt", "Lt", "Xv", "Yv", "Lv"])}
    np.savez(f"{D.W}/train_s{STRIDE}.npz", **out, feats=np.array(FE))
    print({k: v.shape for k, v in out.items()})
