#!/usr/bin/env python3
"""exp1 screen: linear probes -> next-64 normalised salience of layer L (256 experts), standalone sal-hot on heldout
(probe fit on all calib-fit) and calib-val (fit on calib chains %4 != 3, eval chains %4 == 3).
Inputs at block end b (serve-causal: pooled over tokens <= 16(b+1)-1, own layer L):
  st  = v2 state [sema32, sema128] (512)
  lg  = router logits of the pooled context: mean logits over last 16 / 64 tokens, EMA256 (768)
  h   = PCA-512 (calib-fit fit) of the RMS-normalised residual pooled over last 16 / 64 tokens (1024)
  exp1.py [NPROC] [LAYERS]"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
from multiprocessing import Pool  # noqa: E402
import json  # noqa: E402
import numpy as np  # noqa: E402
import hplib as HL  # noqa: E402
import heval as H  # noqa: E402
import t32lib as T  # noqa: E402

LAM = float(os.environ.get("LAM", "1.0"))


def ridge_fit(X, Y, lam=LAM):
    mx, sx, my = X.mean(0), X.std(0) + 1e-6, Y.mean(0)
    Z = (X - mx) / sx
    A = Z.T @ Z
    A[np.diag_indices_from(A)] += lam * len(Z) * 1e-3
    W = np.linalg.solve(A, Z.T @ (Y - my))
    return mx, sx, my, W


def ridge_pred(m, X):
    mx, sx, my, W = m
    return ((X - mx) / sx) @ W + my


def feats(L, corpus, pca):
    d = H.rows(corpus, L)
    h, lg = HL.load_pool(L, corpus)
    lg = lg.astype(np.float32)
    F = {"st": HL.state(d),
         "lg": np.hstack([lg, HL.bmean(lg, 4), HL.bema(lg, 256)])}
    if pca is not None:
        mu, V = pca
        h16 = h.astype(np.float32); h64 = HL.bmean(h16, 4)
        F["h"] = np.hstack([(h16 - mu) @ V, (h64 - mu) @ V])
    return d, F, HL.target(d, L)


ARMS = {"st": ["st"], "st_lg": ["st", "lg"], "st_h": ["st", "h"], "st_lg_h": ["st", "lg", "h"], "lg": ["lg"],
        "h": ["h"]}


def job(L):
    h, _ = HL.load_pool(L, "calib-fit")
    x = HL.bmean(h.astype(np.float32), 4)
    mu = x.mean(0)
    U, s, Vt = np.linalg.svd(x[::2] - mu, full_matrices=False)
    pca = (mu, Vt[:512].T.astype(np.float32))
    dc, Fc, Yc = feats(L, "calib-fit", pca)
    dh, Fh, Yh = feats(L, "glm52-heldout", pca)
    ok = np.isfinite(Yc).all(1)
    cid = HL.chain_id(len(Yc))
    val = cid % 4 == 3
    out = {}
    bs_h, bc_h = dh["bsal"].astype(np.float64), dh["bcnt"].astype(np.float64)
    bs_c, bc_c = dc["bsal"].astype(np.float64), dc["bcnt"].astype(np.float64)
    for arm, parts in ARMS.items():
        Xc = np.hstack([Fc[p] for p in parts]); Xh = np.hstack([Fh[p] for p in parts])
        m = ridge_fit(Xc[ok], Yc[ok])
        S = ridge_pred(m, Xh).astype(np.float32)
        out[("ho", arm)] = H.eval_S(S, L, bs_h, bc_h)
        mv = ridge_fit(Xc[ok & ~val], Yc[ok & ~val])
        Sv = ridge_pred(mv, Xc).astype(np.float32)
        out[("cv", arm)] = H.eval_S(Sv, L, bs_c, bc_c, mask=val)
    # v2 reference on the same calib-val blocks (v2 was trained on all calib-fit: optimistic for v2)
    Sv2 = np.load(f"{H.HP}/private/v2S_glm52-heldout/L{L}.npy")
    out[("ho", "v2")] = H.eval_S(Sv2, L, bs_h, bc_h)
    return L, out


if __name__ == "__main__":
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    Ls = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else T.LAYERS
    with Pool(nproc) as p:
        res = dict(p.map(job, Ls))
    keys = sorted({k for L in res for k in res[L]})
    summ = {}
    for k in keys:
        s = H.summarise({L: res[L][k] for L in Ls})
        summ["/".join(k)] = s
        print(f"{k[0]} {k[1]:10s} sal {s['sal']:6.2f} churn {s['churn']:5.2f}", flush=True)
    json.dump(summ, open(f"{H.HP}/exp1_{os.environ.get('TAGX', 'x')}.json", "w"), indent=1)
