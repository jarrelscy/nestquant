#!/usr/bin/env python3
"""rows_tok: token->routing map features (PRIVATE).  Per layer, map M[v] = smoothed mean top-8 indicator (and mean w^2)
of positions whose input token is v, learned on calib-fit chains out-of-fold (fold = chain % 4; fold f rows use the
map from the other 3 folds; heldout uses folds 0-2 = train only, same as the val fold 3).  At anchor t = 16(b+1)-1:
  tk{k}_cnt = sum_{j=1..k} M_cnt[tok[t+j]],  tk{k}_sal = sum_j M_w2[tok[t+j]] * xn_t / norm_L
tok[t+1] is emitted at the refresh (serve-causal); k>1 uses TRUE future tokens = perfect-draft upper bound of the
lm_head-draft -> token-map route.  NaN where t+k leaves the 2048 window."""
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

KS = (1, 2, 4)
NAMES = [f"tk{k}_{s}" for k in KS for s in ("cnt", "sal")]
A = 4.0


def maps(tok_u, ids, w, sel, U):
    """-> Mc [U,256], Mw [U,256] from positions sel"""
    p = np.nonzero(sel)[0]
    n = np.bincount(tok_u[p], minlength=U).astype(np.float64)
    c = np.zeros((U, 256)); s = np.zeros((U, 256))
    rows = np.repeat(tok_u[p], 8)
    np.add.at(c, (rows, ids[p].astype(np.int64).ravel()), 1.0)
    np.add.at(s, (rows, ids[p].astype(np.int64).ravel()), (w[p].astype(np.float64) ** 2).ravel())
    pc = c.sum(0) / n.sum(); ps = s.sum(0) / n.sum()
    return ((c + A * pc) / (n[:, None] + A)).astype(np.float32), ((s + A * ps) / (n[:, None] + A)).astype(np.float32)


def job(L):
    tokc = T.tokens("calib-fit", 10 ** 9); toh = T.tokens("glm52-heldout", 10 ** 9)
    idc, wc, _ = T.load_layer(L, "calib-fit")
    tokc = tokc[: idc.shape[0]]
    uni, inv = np.unique(np.concatenate([tokc, toh]), return_inverse=True)
    U = len(uni)
    uc, uh = inv[: len(tokc)], inv[len(tokc):]
    fold = (np.arange(len(tokc)) // (T.CHAIN * T.SEQ)) % 4
    mp = {f: maps(uc, idc, wc, fold != f, U) for f in range(4)}
    mp["ho"] = maps(uc, idc, wc, fold != 3, U) if 3 in mp else None
    for corpus in ("calib-fit", "glm52-heldout"):
        d = D.load_rows(corpus, L)
        bc, bs, cand = d["bcnt"].astype(np.float64), d["bsal"].astype(np.float64), d["cand"].astype(np.int64)
        nb = bc.shape[0]
        nrm = norm_of(bc, bs)
        ids, w, xn = T.load_layer(L, corpus)
        tu = uc if corpus == "calib-fit" else uh[: ids.shape[0]]
        anc = (np.arange(nb) + 1) * T.G - 1
        xa = xn[anc].astype(np.float64)
        bf = (anc // (T.CHAIN * T.SEQ)) % 4 if corpus == "calib-fit" else np.full(nb, 3)
        F = {}
        for k in KS:
            ok = (anc % T.SEQ) + k < T.SEQ
            cnt = np.zeros((nb, 256)); ws = np.zeros((nb, 256))
            for f in range(4):
                b = np.nonzero(ok & (bf == f))[0]
                Mc, Mw = mp[f]
                for j in range(1, k + 1):
                    v = tu[anc[b] + j]
                    cnt[b] += Mc[v]; ws[b] += Mw[v]
            sal = ws * xa[:, None] / nrm
            cnt[~ok] = np.nan; sal[~ok] = np.nan
            F[f"tk{k}_cnt"], F[f"tk{k}_sal"] = cnt, sal
        Fa = np.stack([np.take_along_axis(F[n], cand, 1) for n in NAMES], -1).astype(np.float32)
        os.makedirs(f"{D.PRIV}/rows_tok/{corpus}", exist_ok=True)
        np.savez(f"{D.PRIV}/rows_tok/{corpus}/L{L}.npz", F=Fa)
    return L, U


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] != "all":
        print(job(int(sys.argv[1])))
    else:
        with Pool(12) as p:
            print(len(p.map(job, T.LAYERS)))
