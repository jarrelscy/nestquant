#!/usr/bin/env python3
"""gbz.py ARM [k]: v2-family LightGBM (tweedie 1.5, lr .3, 15 leaves, min_data 500, bag .5, ff .9, <=60 trees) on
calib-fit train chains (early stop on calib-val chains), features = v2's 9 (+ arm extras), target next-64 sal / m_L.
ARM: base (v2 feats, my split) | zpca (+ k global PCA comps of all layers' lsema128|lsema512, causal) |
     zown (+ k own-layer PCA comps) | znn:NAME (+ z of trained XLatent NAME, from D/<c>/z_NAME.npy)
-> scores D/<c>/S_gbz_ARM.npy for calib val + heldout, report hm 0.3/0.5/0.7."""
import json, os, sys, time
import numpy as np
import xlib as X

arm = sys.argv[1]; k = int(sys.argv[2]) if len(sys.argv) > 2 else 8
tag = arm.replace(":", "_") + (f"{k}" if arm in ("zpca", "zown") else "")
FX, FD = X.masks()
sg = X.chains("calib-fit")
VAL = [4, 9, 14, 19, 24, 29]
TR = [i for i in range(len(sg)) if i not in VAL]
bs_c = X.load("calib-fit", "bsal", mmap=False)
mL = bs_c.sum((0, 2), dtype=np.float64) / (bs_c.shape[0] * 16 * 8)
t0 = time.time()


def y64(bsal, sgs):
    Y = np.full(bsal.shape, np.nan, np.float32)
    for s, e in sgs:
        b = np.asarray(bsal[s:e], np.float64) / mL[None, :, None]
        cs = np.concatenate([np.zeros((1,) + b.shape[1:]), np.cumsum(b, 0)])
        n = e - s; kk = np.arange(n - 4)
        Y[s + kk] = cs[kk + 5] - cs[kk + 1]
    return Y


def gstate(F):
    return np.concatenate([np.asarray(F[..., 2], np.float32).reshape(F.shape[0], -1),
                           np.asarray(F[..., 3], np.float32).reshape(F.shape[0], -1)], 1)


def zfeat(corpus):
    """-> [nb, 75, kz] extra per-(block, layer) features (broadcast over experts)"""
    if arm == "base":
        return None
    if arm.startswith("znn:"):
        z = np.load(f"{X.D}/{corpus}/z_{arm[4:]}.npy")                       # [nb, dz]
        return np.broadcast_to(z[:, None, :k], (z.shape[0], 75, min(k, z.shape[1])))
    cache = f"{X.D}/pca_{arm}.npz"
    if not os.path.exists(cache):
        Gc = gstate(X.load("calib-fit", "feat7"))
        tr = np.concatenate([np.arange(*sg[i]) for i in TR])
        mu = Gc[tr].mean(0)
        if arm == "zpca":
            A = Gc[tr[::2]] - mu
            w, V = np.linalg.eigh(A @ A.T)                                      # Gram trick: n x n
            V = (A.T @ V[:, ::-1][:, :64]) / np.sqrt(np.maximum(w[::-1][:64], 1e-9))
            np.savez(cache, mu=mu, P=V.T.astype(np.float32))
        else:
            Ps = []
            for i in range(75):
                cols = np.r_[i * 256:(i + 1) * 256, 19200 + i * 256:19200 + (i + 1) * 256]
                _, _, Vi = np.linalg.svd(Gc[tr[::2]][:, cols] - mu[cols], full_matrices=False)
                Ps.append(Vi[:16])
            np.savez(cache, mu=mu, P=np.stack(Ps).astype(np.float32))
        print("pca", time.time() - t0, flush=True)
    z = np.load(cache)
    G = gstate(X.load(corpus, "feat7")) - z["mu"]
    if arm == "zpca":
        Z = G @ z["P"][:k].T
        return np.broadcast_to(Z[:, None, :], (Z.shape[0], 75, k))
    out = np.zeros((G.shape[0], 75, k), np.float32)
    for i in range(75):
        cols = np.r_[i * 256:(i + 1) * 256, 19200 + i * 256:19200 + (i + 1) * 256]
        out[:, i] = G[:, cols] @ z["P"][i, :k].T
    return out


def rows(corpus, blocks, Zx, Y=None, sub=1):
    F9 = X.load(corpus, "F9")
    b = blocks[::sub]
    nfm = ~FX
    Xb = np.asarray(F9[b], np.float32)[:, nfm]                                  # [n, P, 9]  (P = non-fixed slots)
    parts = [Xb]
    if Zx is not None:
        li = np.nonzero(nfm)[0]
        parts.append(np.asarray(Zx[b], np.float32)[:, li])
    M = np.concatenate(parts, -1).reshape(-1, sum(p.shape[-1] for p in parts))
    y = None if Y is None else Y[b][:, nfm].ravel()
    return M, y


import lightgbm as lgb
Zc = zfeat("calib-fit"); Zh = zfeat("glm52-heldout")
Yc = y64(bs_c, sg)
tr = np.concatenate([np.arange(*sg[i]) for i in TR]); va = np.concatenate([np.arange(*sg[i]) for i in VAL])
trv = tr[np.isfinite(Yc[tr, 0, 0])]; vav = va[np.isfinite(Yc[va, 0, 0])]
Xt, yt = rows("calib-fit", trv, Zc, Yc, sub=3)
Xv, yv = rows("calib-fit", vav, Zc, Yc, sub=3)
print(arm, k, "rows", Xt.shape, Xv.shape, f"{time.time() - t0:.0f}s", flush=True)
p = dict(objective="tweedie", metric="tweedie", tweedie_variance_power=1.5, learning_rate=0.3, num_leaves=15,
         min_data_in_leaf=500, bagging_fraction=0.5, bagging_freq=1, bagging_seed=3, feature_fraction=0.9, seed=0,
         num_threads=int(os.environ.get("NT", "20")), verbosity=-1, max_bin=255)
dt = lgb.Dataset(Xt, yt); dv = lgb.Dataset(Xv, yv, reference=dt)
bst = lgb.train(p, dt, 60, valid_sets=[dv], callbacks=[lgb.early_stopping(10, verbose=False)])
os.makedirs("/tmp/nestquant/33-search/xlatent/models", exist_ok=True)
bst.save_model(f"/tmp/nestquant/33-search/xlatent/models/gbz_{tag}.txt")
print("trained", bst.best_iteration, f"{time.time() - t0:.0f}s", flush=True)
res = dict(arm=arm, k=k, best_iter=bst.best_iteration)


def score(corpus, blocks, Zx):
    S = np.zeros((len(blocks), 75, 256), np.float32)
    nfm = ~FX
    for c0 in range(0, len(blocks), 512):
        bb = blocks[c0:c0 + 512]
        M, _ = rows(corpus, bb, Zx)
        pr = bst.predict(M, num_iteration=bst.best_iteration, num_threads=p["num_threads"])
        tmp = np.zeros((len(bb), 75, 256), np.float32); tmp[:, nfm] = pr.reshape(len(bb), -1)
        S[c0:c0 + 512] = tmp
    return S


vs = [sg[i] for i in VAL]
Sv = np.zeros((bs_c.shape[0], 75, 256), np.float32); Sv[va] = score("calib-fit", va, Zc)
for hm in (0.5,):
    m = X.metrics(X.replay(Sv, FX, FD, vs, hm), bs_c, FX, vs, tf=True)
    res[f"val_hm{hm}"] = dict(sal=m["sal"], churn=m["churn"])
    print(f"calib-val gbz_{tag} hm{hm}: {m['sal'] * 100:.2f} churn {m['churn']:.2f}", flush=True)
nbh = X.load("glm52-heldout", "bsal").shape[0]
Sh = score("glm52-heldout", np.arange(nbh), Zh)
np.save(f"{X.D}/glm52-heldout/S_gbz_{tag}.npy", Sh)
for hm in (0.3, 0.4, 0.5, 0.7):
    m = X.evaluate(Sh, "glm52-heldout", hm)
    res[f"ho_hm{hm}"] = {kk: v for kk, v in m.items() if kk != "per_layer"}
    print(f"heldout gbz_{tag} hm{hm}: {m['sal'] * 100:.2f} churn {m['churn']:.2f} L3-6 {m['L3_6'] * 100:.1f} "
          f"L7-40 {m['L7_40'] * 100:.1f} L41-77 {m['L41_77'] * 100:.1f}", flush=True)
json.dump(res, open(f"/tmp/nestquant/33-search/xlatent/gbz_{tag}.json", "w"), indent=1)
