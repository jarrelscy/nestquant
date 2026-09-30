#!/usr/bin/env python3
"""rows_draft: GBDT features from the MTP-draft routing predictor (fit_draft.py private/pred) at the refresh anchor
t = 16(b+1)-1, for the next k = 1, 2, 4 positions (t+1 .. t+k), band-all cand order.  PRIVATE.
  d{k}_cnt   sum_j [e in predicted top-8 of position t+j]          (hits analogue of mtp{k}_cnt)
  d{k}_sal   sum_j predicted w^2 * xn_t / norm_L  (w from the predicted logits' top-8, xn of the current token)
  d{k}_soft  sum_j sigmoid(predicted logit)                        (graded, informative near the top-8 cutoff)
and the same-recipe ORACLE (true routing of t+1..t+k from the trace; = T32 mtp{k}, window-bounded like the drafts):
  o{k}_cnt, o{k}_sal
NaN where t+j leaves the 2048 window (independent capture windows)."""
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
out = f"{D.PRIV}/rows_draft/{corpus}"
KS = (1, 2, 4)
NAMES = [f"d{k}_{s}" for k in KS for s in ("cnt", "sal", "soft")] + [f"o{k}_{s}" for k in KS for s in ("cnt", "sal")]


def job(L):
    d = D.load_rows(corpus, L)
    bc, bs, cand = d["bcnt"].astype(np.float64), d["bsal"].astype(np.float64), d["cand"].astype(np.int64)
    nb = bc.shape[0]
    nrm = norm_of(bc, bs)
    P = np.load(f"{D.PRIV}/pred/{corpus}/L{L}.npz")["P"].astype(np.float32)[:nb]     # [nb, 4, 256]
    bias = np.load(f"{T.OUT}/trace2/L{L}.r0of8.npz")["bias"].astype(np.float32)
    ids, w, xn = T.load_layer(L, corpus)
    anc = (np.arange(nb) + 1) * T.G - 1
    xa = xn[anc].astype(np.float64)
    sg = 1 / (1 + np.exp(-P))                                  # [nb, 4, 256]
    sel = sg + bias
    ok = np.isfinite(P[..., 0])                                # [nb, 4]
    top = np.argsort(-np.where(np.isfinite(sel), sel, -np.inf), -1)[..., :8]
    ind = np.zeros(P.shape, bool)
    np.put_along_axis(ind, top, True, -1)
    ind &= ok[..., None]
    wt = np.where(ind, sg, 0)
    wt = wt / np.maximum(wt.sum(-1, keepdims=True), 1e-20) * 2.5
    v = w.astype(np.float64) ** 2 * xn.astype(np.float64)[:, None]
    F = {}
    for k in KS:
        m = ok[:, k - 1]
        cnt = ind[:, :k].sum(1).astype(np.float64)
        sal = ((wt[:, :k] ** 2).sum(1) * xa[:, None]) / nrm
        soft = np.nan_to_num(sg[:, :k]).sum(1)
        for n, a in (("cnt", cnt), ("sal", sal), ("soft", soft)):
            a = a.copy(); a[~m] = np.nan
            F[f"d{k}_{n}"] = a
        oc, osl = np.zeros((nb, 256)), np.zeros((nb, 256))
        for j in range(1, k + 1):
            t = anc + j
            b = np.nonzero(m & (t < ids.shape[0]))[0]
            np.add.at(oc, (np.repeat(b, 8), ids[t[b]].astype(np.int64).ravel()), 1.0)
            np.add.at(osl, (np.repeat(b, 8), ids[t[b]].astype(np.int64).ravel()), v[t[b]].ravel())
        oc[~m] = np.nan; osl[~m] = np.nan
        F[f"o{k}_cnt"], F[f"o{k}_sal"] = oc, osl / nrm
    Fa = np.stack([np.take_along_axis(F[n], cand, 1) for n in NAMES], -1).astype(np.float32)
    os.makedirs(out, exist_ok=True)
    np.savez(f"{out}/L{L}.npz", F=Fa)
    return L


if __name__ == "__main__":
    with Pool(16) as p:
        print(len(p.map(job, T.LAYERS)))
