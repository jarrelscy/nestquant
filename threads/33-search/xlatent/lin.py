#!/usr/bin/env python3
"""lin.py: linear diagnostic of the factorised shared-latent model on v2's log-residual.
  R[b,L,e] = log(Y64 + c) - log(S_v2 + c);  R ~ b_Le + U_Le . z(b),  z = PCA_k(all layers' lsema128|lsema512)
arms: bias-only, shared-z (k), own-layer-z (per layer PCA), shared-z + own; shrink s in {0.25,0.5,1}."""
import json, sys, time
import numpy as np
import xlib as X

C = 0.1
FX, FD = X.masks()
nf = ~FX
bs_c = X.load("calib-fit", "bsal", mmap=False)
mL = bs_c.sum((0, 2), dtype=np.float64) / (bs_c.shape[0] * 16 * 8)
sg = X.chains("calib-fit")
VAL = [4, 9, 14, 19, 24, 29]
TR = [i for i in range(len(sg)) if i not in VAL]


def y64(bsal, sgs):
    Y = np.full(bsal.shape, np.nan, np.float32)
    for s, e in sgs:
        b = np.asarray(bsal[s:e], np.float64) / mL[None, :, None]
        cs = np.concatenate([np.zeros((1,) + b.shape[1:]), np.cumsum(b, 0)])
        n = e - s; k = np.arange(n - 4)
        Y[s + k] = cs[k + 5] - cs[k + 1]
    return Y


def gstate(F):
    """global per-block state: [nb, 75*256*2] lsema128 | lsema512"""
    return np.concatenate([np.asarray(F[..., 2], np.float32).reshape(F.shape[0], -1),
                           np.asarray(F[..., 3], np.float32).reshape(F.shape[0], -1)], 1)


t0 = time.time()
Fc = X.load("calib-fit", "feat7"); Fh = X.load("glm52-heldout", "feat7")
Yc = y64(bs_c, sg)
Sc = X.load("calib-fit", "S_v2", mmap=False); Sh = X.load("glm52-heldout", "S_v2", mmap=False)
Rc = np.log(Yc + C) - np.log(np.maximum(Sc, 0) + C)
tr = np.concatenate([np.arange(*sg[i]) for i in TR]); va = np.concatenate([np.arange(*sg[i]) for i in VAL])
trv = tr[np.isfinite(Rc[tr, 0, 0])]
Gc, Gh = gstate(Fc), gstate(Fh)
mu = Gc[tr].mean(0)
# PCA on train blocks
A = Gc[trv] - mu
_, sv, Vt = np.linalg.svd(A[::2], full_matrices=False)
print("pca var share top8/32/64", [float((sv[:k] ** 2).sum() / (sv ** 2).sum()) for k in (8, 32, 64)],
      f"{time.time() - t0:.0f}s", flush=True)


def fit_pred(Z_tr, R_tr, Z_list, lam=1.0):
    """ridge with intercept per output column; R_tr [n, P]. -> predictions for each Z in Z_list [m, P]"""
    Z1 = np.c_[np.ones(len(Z_tr)), Z_tr]
    Rg = lam * np.eye(Z1.shape[1]); Rg[0, 0] = 0
    W = np.linalg.solve(Z1.T @ Z1 + Rg, Z1.T @ R_tr)
    return [np.c_[np.ones(len(Z)), Z] @ W for Z in Z_list]


res = {}
Rtr = Rc[trv].reshape(len(trv), -1)
Rva = Rc[va].reshape(len(va), -1); vv = np.isfinite(Rva[:, 0])
nfm = np.broadcast_to(nf.ravel(), Rtr.shape[1:])


def report(name, Pc_va, Ph):
    r2 = 1 - np.nanmean(((Rva - Pc_va)[vv][:, nfm]) ** 2) / np.nanmean(((Rva - Rtr.mean(0))[vv][:, nfm]) ** 2)
    out = dict(val_r2_vs_bias=float(r2))
    for s in (0.5, 1.0):
        Sv = Sc.copy()
        Sv[va] = (np.maximum(Sc[va], 0) + C) * np.exp(s * Pc_va.reshape(-1, 75, 256)) - C
        Shh = (np.maximum(Sh, 0) + C) * np.exp(s * Ph.reshape(-1, 75, 256)) - C
        mv = X.metrics(X.replay(Sv, FX, FD, [sg[i] for i in VAL]), bs_c, FX, [sg[i] for i in VAL], tf=True)
        mh = X.evaluate(Shh.astype(np.float32), "glm52-heldout")
        out[f"s{s}"] = dict(val=mv["sal"], val_churn=mv["churn"], ho=mh["sal"], ho_churn=mh["churn"])
    res[name] = out
    print(name, json.dumps(out), f"{time.time() - t0:.0f}s", flush=True)


mv0 = X.metrics(X.replay(Sc, FX, FD, [sg[i] for i in VAL]), bs_c, FX, [sg[i] for i in VAL], tf=True)
print("v2 val", mv0["sal"], mv0["churn"], flush=True)
# bias only
b = Rtr.mean(0)
report("bias", np.broadcast_to(b, Rva.shape), np.broadcast_to(b, (Gh.shape[0], b.size)))
for k in [int(x) for x in (sys.argv[1:] or ["8", "32", "64"])]:
    P = Vt[:k].T
    Ztr, Zva, Zh = (Gc[trv] - mu) @ P, (Gc[va] - mu) @ P, (Gh - mu) @ P
    sd = Ztr.std(0); Ztr, Zva, Zh = Ztr / sd, Zva / sd, Zh / sd
    for lam in (10.0, 1000.0):
        pv, ph = fit_pred(Ztr, Rtr, [Zva, Zh], lam)
        report(f"shared_k{k}_lam{lam:g}", pv, ph)
# own-layer z (8 per layer, PCA of own 512-dim state) for comparison
kk = 8
pv = np.zeros_like(Rva); ph = np.zeros((Gh.shape[0], Rva.shape[1]), np.float32)
for i in range(75):
    cols = np.r_[i * 256:(i + 1) * 256, 19200 + i * 256:19200 + (i + 1) * 256]
    Ai = Gc[trv][:, cols] - mu[cols]
    _, _, Vi = np.linalg.svd(Ai[::2], full_matrices=False)
    P = Vi[:kk].T
    Zt, Zv, Zhh = Ai @ P, (Gc[va][:, cols] - mu[cols]) @ P, (Gh[:, cols] - mu[cols]) @ P
    sd = Zt.std(0)
    o = slice(i * 256, (i + 1) * 256)
    a_, b_ = fit_pred(Zt / sd, Rtr[:, o], [Zv / sd, Zhh / sd], 10.0)
    pv[:, o], ph[:, o] = a_, b_
report("own_k8", pv, ph)
json.dump(res, open("/tmp/nestquant/33-search/xlatent/lin.json", "w"), indent=1)
