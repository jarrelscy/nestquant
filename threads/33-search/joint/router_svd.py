"""T33i jF-H/jF-R prep (CPU): fixed low-rank residual basis = top right-singular vectors of the stacked (per-layer
post_attention_layernorm-weighted) router matrices W_r[L] diag(ln_L), L=3..77 -> hproj/router_svd.npz (V [6144, 512],
s).  Diagnostic: on T32 hidcov (residual leaving L10/40/70, RMS-normalised, calib-fit; aggregate cov only) the share of
next-layer router-logit variance and of residual variance captured by router-SVD-k vs PCA-k.  PRIVATE."""
import glob
import numpy as np
import torch

O = "/tmp/nestquant/33-search/joint"
torch.set_num_threads(16)
z = np.load(f"{O}/router_w.npz"); W = z["W"].astype(np.float32); ln = z["ln"]; lay = list(z["layers"])
M = torch.from_numpy((W * ln[:, None, :]).reshape(-1, W.shape[-1]))            # [19200, 6144]
G = (M.T.double() @ M.double())                                                  # [6144, 6144]
ev, V = torch.linalg.eigh(G); o = torch.argsort(ev, descending=True); ev, V = ev[o], V[:, o]
s = ev.clamp(min=0).sqrt()
np.savez(f"{O}/hproj/router_svd.npz", V=V[:, :512].float().numpy(), s=s[:512].float().numpy())
e = ev.clamp(min=0) / ev.clamp(min=0).sum()
print("router stack energy in top-k:", {k: round(float(e[:k].sum()), 3) for k in (32, 64, 128, 256, 512, 1024)})
for Lh in (10, 40, 70):
    n = 0; S1 = SS = None
    for f in sorted(glob.glob(f"/tmp/nestquant/32-gbdt-sal/private/hidcov/hidcov_L{Lh}.r*.npz")):
        q = np.load(f); n += int(q["n"]); S1 = q["s"] if S1 is None else S1 + q["s"]; SS = q["ss"] if SS is None else SS + q["ss"]
    mu = S1 / n; C = torch.from_numpy(SS / n - np.outer(mu, mu))                  # f64 [6144, 6144]
    ce, CV = torch.linalg.eigh(C); o = torch.argsort(ce, descending=True); CV = CV[:, o]
    Ln = Lh + 1; i = lay.index(Ln)
    A = torch.from_numpy(W[i] * ln[i][None]).double()                            # next layer's router on this residual
    tot = torch.trace(A @ C @ A.T)
    row = {}
    for k in (32, 128, 256, 512):
        for nm, B in (("rsvd", V[:, :k]), ("pca", CV[:, :k])):
            P = B @ B.T
            row[f"{nm}{k}"] = (round(float(torch.trace(A @ P @ C @ P @ A.T) / tot), 3), round(float(torch.trace(P @ C) / torch.trace(C)), 3))
    # own-layer router row space (rank <= 256): upper bound for logit variance, residual share
    Q, _ = torch.linalg.qr(A.T); P = Q @ Q.T
    row["own256"] = (1.0, round(float(torch.trace(P @ C) / torch.trace(C)), 3))
    print(f"resid L{Lh} (n {n}) -> router L{Ln}: (logit-var share, resid-var share)", row, flush=True)
