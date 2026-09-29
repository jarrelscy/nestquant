#!/usr/bin/env python3
"""T32 cross-layer lead-lag (Granger-style) test: does routing drift show up in some layer bands before others?
Per layer L, per 16-token block t (within chain), salience share histogram s_L(t) (bsal row-normalised):
  recent change  r_L(t) = mean s_L(t-3..t) - mean s_L(t-7..t-4)                (last 64 vs previous 64 tokens)
  future change  f_L(t) = mean s_L(t+a..t+b) - mean s_L(t-7..t)                 (near: blocks t+1..t+2;
                                                                                 far: t+5..t+8; all: t+1..t+8)
Per layer PCA-16 bases of r and f fitted on calib-fit.  For target layer L in band B, ridge regressions (fit calib-fit,
R^2 on glm52-heldout, of the 16 f-components, variance-weighted):
  own     : r of layer L itself
  own+A   : + r of every layer in band A (A != B: other band; A = B: the rest of B)
  A only  : r of band A layers only
lead of A on B = R^2(own+A) - R^2(own)  (> 0 = A's recent change carries information about B's future drift beyond
B's own recent change).  Output $OUT/leadlag.json + table."""
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np  # noqa: E402

import t32lib as T  # noqa: E402

NBC = T.CHAIN * T.SEQ // T.G
BANDS = {"L3-6": range(3, 7), "L7-20": range(7, 21), "L21-40": range(21, 41), "L41-60": range(41, 61),
         "L61-77": range(61, 78)}
HZ = {"near": (1, 2), "far": (5, 8), "all": (1, 8)}
K = 16
LAM = float(os.environ.get("LAM", "10.0"))


def shares(corpus, L):
    s = np.load(f"{T.OUT}/rows_bandall/{corpus}/L{L}.npz")["bsal"].astype(np.float64)
    tot = s.sum(1, keepdims=True)
    return np.where(tot > 0, s / np.maximum(tot, 1e-30), 1.0 / T.NE)


def win_mean(S, a, b):
    """per chain: mean of S over blocks t+a .. t+b (inclusive), NaN outside the chain."""
    out = np.full(S.shape, np.nan)
    for c0 in range(0, S.shape[0], NBC):
        x = S[c0:c0 + NBC]
        cs = np.vstack([np.zeros((1, S.shape[1])), np.cumsum(x, 0)])
        t = np.arange(x.shape[0])
        lo, hi = t + a, t + b
        ok = (lo >= 0) & (hi < x.shape[0])
        m = np.full(x.shape, np.nan)
        m[ok] = (cs[hi[ok] + 1] - cs[lo[ok]]) / (b - a + 1)
        out[c0:c0 + NBC] = m
    return out


def changes(corpus, L):
    s = shares(corpus, L)
    r = win_mean(s, -3, 0) - win_mean(s, -7, -4)
    past = win_mean(s, -7, 0)
    f = {h: win_mean(s, a, b) - past for h, (a, b) in HZ.items()}
    return r, f


def pca(X):
    m = np.isfinite(X).all(1)
    mu = X[m].mean(0)
    _, _, Vt = np.linalg.svd(X[m] - mu, full_matrices=False)
    return mu, Vt[:K].T


def ridge_r2(Xtr, Ytr, Xte, Yte):
    mx, my = Xtr.mean(0), Ytr.mean(0)
    sx = Xtr.std(0) + 1e-12
    A = (Xtr - mx) / sx; B = (Xte - mx) / sx
    W = np.linalg.solve(A.T @ A + LAM * len(A) / 1000 * np.eye(A.shape[1]), A.T @ (Ytr - my))
    P = B @ W + my
    return 1.0 - ((Yte - P) ** 2).sum() / ((Yte - Yte.mean(0)) ** 2).sum()


if __name__ == "__main__":
    R = {c: {} for c in ("calib-fit", "glm52-heldout")}
    F = {c: {} for c in R}
    for L in T.LAYERS:
        rc, fc = changes("calib-fit", L)
        mu_r, V_r = pca(rc)
        R["calib-fit"][L] = (rc - mu_r) @ V_r
        rh, fh = changes("glm52-heldout", L)
        R["glm52-heldout"][L] = (rh - mu_r) @ V_r
        for h in HZ:
            mu_f, V_f = pca(fc[h])
            F["calib-fit"].setdefault(h, {})[L] = (fc[h] - mu_f) @ V_f
            F["glm52-heldout"].setdefault(h, {})[L] = (fh[h] - mu_f) @ V_f
        print(L, end=" ", flush=True)
    print()

    def valid(c, h):
        m = np.ones(R[c][T.LAYERS[0]].shape[0], bool)
        for L in T.LAYERS:
            m &= np.isfinite(R[c][L]).all(1) & np.isfinite(F[c][h][L]).all(1)
        return m

    res = {}
    for h in HZ:
        mtr, mte = valid("calib-fit", h), valid("glm52-heldout", h)
        for bn, bl in BANDS.items():
            for an, al in BANDS.items():
                own, both, only = [], [], []
                for L in bl:
                    src = [J for J in al if J != L]
                    Ytr, Yte = F["calib-fit"][h][L][mtr], F["glm52-heldout"][h][L][mte]
                    Xo_tr, Xo_te = R["calib-fit"][L][mtr], R["glm52-heldout"][L][mte]
                    Xa_tr = np.hstack([R["calib-fit"][J][mtr] for J in src])
                    Xa_te = np.hstack([R["glm52-heldout"][J][mte] for J in src])
                    own.append(ridge_r2(Xo_tr, Ytr, Xo_te, Yte))
                    both.append(ridge_r2(np.hstack([Xo_tr, Xa_tr]), Ytr, np.hstack([Xo_te, Xa_te]), Yte))
                    only.append(ridge_r2(Xa_tr, Ytr, Xa_te, Yte))
                res[f"{h}|{an}->{bn}"] = dict(own=float(np.mean(own)), own_plus_A=float(np.mean(both)),
                                              A_only=float(np.mean(only)), lead=float(np.mean(both) - np.mean(own)))
            print(h, bn, "own R2 %.4f" % res[f"{h}|{bn}->{bn}"]["own"], flush=True)
    json.dump(res, open(f"{T.OUT}/leadlag.json", "w"), indent=1)
    for h in HZ:
        print(f"\n[{h}] lead = R2(own+A) - R2(own) on heldout (rows: source A, cols: target B); own R2 on the diagonal "
              f"line below")
        print(" " * 8 + "".join(f"{b:>10s}" for b in BANDS))
        for an in BANDS:
            print(f"{an:8s}" + "".join(f"{res[f'{h}|{an}->{bn}']['lead']:10.4f}" for bn in BANDS))
        print(f"{'own':8s}" + "".join(f"{res[f'{h}|{bn}->{bn}']['own']:10.4f}" for bn in BANDS))
        print(f"{'A-only':8s} (diag) " + " ".join(f"{bn} {res[f'{h}|{bn}->{bn}']['A_only']:.4f}" for bn in BANDS))
