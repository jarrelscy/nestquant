#!/usr/bin/env python3
"""build per-(layer, expert) probe features [nb, 256] for calib-fit (out-of-fold, 4 chain folds chain%4) and
glm52-heldout (fit on all calib-fit) -> private/feat/{NAME}/{corpus}/L{L}.npy; standalone eval of ridge probes.
  make_feats.py SET [NPROC]
SETs:
  lgraw   lg16 lg64 lgd (= lg64 - lgE256): router logits of the pooled context (no fit)
  lgpr    ridge probes: pr_st [state], pr_stlg [state, lg16, lg64, lgE256]
  hpr     ridge probes: pr_sth [state, PCA-K h16, h64], pr_stlgh [+ lg]  (K = env PCAK, default 512)
  dnpr    downstream routers: pooled residual of L through routers L..L+4 (norm weight of target layer)
Ridge target: next-64 normalised salience (v2 target), by expert id."""
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
from multiprocessing import Pool  # noqa: E402
import numpy as np  # noqa: E402
import heval as H  # noqa: E402
import hplib as HL  # noqa: E402
import t32lib as T  # noqa: E402

LAM = float(os.environ.get("LAM", "1.0"))
PCAK = int(os.environ.get("PCAK", "512"))
LGSRC = os.environ.get("LGSRC", "pool")
TGT = os.environ.get("TGT", "raw")                      # raw | log (ridge on log1p target, log1p state inputs)
SFX = "" if TGT == "raw" else "_log"                 # pool (GPU capture) | lg2 (trace2-derived)
FD = f"{H.HP}/private/feat"
CORP = ("calib-fit", "glm52-heldout")


def ridge_fit(X, Y, lam=LAM):
    mx, sx, my = X.mean(0), X.std(0) + 1e-6, Y.mean(0)
    Z = (X - mx) / sx
    A = Z.T @ Z
    A[np.diag_indices_from(A)] += lam * len(Z) * 1e-3
    return mx, sx, my, np.linalg.solve(A, Z.T @ (Y - my))


def ridge_pred(m, X):
    mx, sx, my, W = m
    return ((X - mx) / sx) @ W + my


def get_lg(L, c):
    if LGSRC == "lg2":
        return np.load(f"{H.HP}/private/lg2/{c}/L{L}.npy")
    return HL.load_pool(L, c, keys=("lg",))[0].astype(np.float32)


def get_h(L, c):
    return HL.load_pool(L, c, keys=("h",))[0].astype(np.float32)


def save(name, c, L, A):
    os.makedirs(f"{FD}/{name}/{c}", exist_ok=True)
    np.save(f"{FD}/{name}/{c}/L{L}.npy", A.astype(np.float32))


def fit_oof(Xc, Yc, Xh, Ys=None):
    """-> calib OOF preds (4 chain folds), heldout preds (fit on all).  Gram shared across folds (standardisation
    from all valid calib rows, unsupervised); Ys: optional list of extra targets -> list of (Pc, Ph)."""
    ok = np.isfinite(Yc).all(1)
    cid = HL.chain_id(len(Yc))
    fold = cid * 4 // (cid.max() + 1) if os.environ.get("FOLD") == "contig" else cid % 4   # contig: 4 blocks of chains
    mx, sx = Xc[ok].mean(0), Xc[ok].std(0) + 1e-6
    Z = ((Xc - mx) / sx).astype(np.float64); Zh = (Xh - mx) / sx
    Zf = np.hstack([Z, np.ones((len(Z), 1))])           # intercept column (unpenalised)
    Gf = [Zf[ok & (fold == f)].T @ Zf[ok & (fold == f)] for f in range(4)]
    out = []
    for Y in [Yc] + (Ys or []):
        Y = np.where(np.isfinite(Y), Y, 0.0)
        Bf = [Zf[ok & (fold == f)].T @ Y[ok & (fold == f)] for f in range(4)]
        def solve(G, B, n):
            A = G.copy(); A[np.diag_indices(A.shape[0] - 1)] += LAM * n * 1e-3
            return np.linalg.solve(A, B)
        Pc = np.empty(Y.shape, np.float32)
        for f in range(4):
            idx = [g for g in range(4) if g != f]
            W = solve(sum(Gf[g] for g in idx), sum(Bf[g] for g in idx), int((ok & (fold != f)).sum()))
            Pc[fold == f] = Zf[fold == f] @ W
        W = solve(sum(Gf), sum(Bf), int(ok.sum()))
        out.append((Pc, (np.hstack([Zh, np.ones((len(Zh), 1))]) @ W).astype(np.float32)))
    return out if Ys else out[0]


def job(args):
    S_, L = args
    d = {c: H.rows(c, L) for c in CORP}
    Y = HL.target(d["calib-fit"], L)
    if TGT == "log":
        Y = np.log1p(Y)
    out = {}
    if S_ == "lgraw":
        for c in CORP:
            lg = get_lg(L, c)
            l64 = HL.bmean(lg, 4)
            save("lg16", c, L, lg); save("lg64", c, L, l64); save("lgd", c, L, l64 - HL.bema(lg, 256))
        return L, out
    st = {c: HL.state(d[c]) for c in CORP}
    if TGT == "log":
        st = {c: np.log1p(v) for c, v in st.items()}
    ins = {}
    if S_ in ("lgpr", "hpr"):
        ins["st"] = st
    if S_ == "lgpr" or S_ == "hpr":
        lgf = {}
        for c in CORP:
            lg = get_lg(L, c)
            lgf[c] = np.hstack([lg, HL.bmean(lg, 4), HL.bema(lg, 256)])
        ins["lg"] = lgf
    if S_ == "hpr":
        hs = {c: get_h(L, c) for c in CORP}
        x = HL.bmean(hs["calib-fit"], 4)
        mu = x.mean(0)
        C = np.cov((x[::2] - mu).T.astype(np.float64), bias=True) if False else None
        xs = (x[::2] - mu).astype(np.float32)
        rng = np.random.default_rng(L)
        Q = np.linalg.qr(xs.T @ (xs @ rng.standard_normal((xs.shape[1], PCAK + 64)).astype(np.float32)))[0]
        Q = np.linalg.qr(xs.T @ (xs @ Q))[0]
        _, _, Vt = np.linalg.svd(xs @ Q, full_matrices=False)
        V = (Q @ Vt.T)[:, :PCAK].astype(np.float32)
        ins["h"] = {c: np.hstack([(hs[c] - mu) @ V, (HL.bmean(hs[c], 4) - mu) @ V]) for c in CORP}
        del hs
    arms = {"lgpr": {"pr_st": ["st"], "pr_stlg": ["st", "lg"]},
            "hpr": {"pr_st": ["st"], f"pr_sth{PCAK}": ["st", "h"], f"pr_stlgh{PCAK}": ["st", "lg", "h"]}}[S_]
    for name, parts in arms.items():
        name = name + SFX
        Xc = np.hstack([ins[p]["calib-fit"] for p in parts]); Xh = np.hstack([ins[p]["glm52-heldout"] for p in parts])
        Pc, Ph = fit_oof(Xc, Y, Xh)
        save(name, "calib-fit", L, Pc); save(name, "glm52-heldout", L, Ph)
        dh = d["glm52-heldout"]
        out[name] = H.eval_S(Ph, L, dh["bsal"].astype(np.float64), dh["bcnt"].astype(np.float64))
        dc = d["calib-fit"]
        out[name + "|cv"] = H.eval_S(Pc, L, dc["bsal"].astype(np.float64), dc["bcnt"].astype(np.float64),
                                     mask=HL.chain_id(len(Pc)) % 4 == 3)
    return L, out


if __name__ == "__main__":
    S_ = sys.argv[1]
    nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    with Pool(nproc) as p:
        res = dict(p.map(job, [(S_, L) for L in T.LAYERS]))
    names = sorted(res[T.LAYERS[0]])
    summ = {}
    for n in names:
        s = H.summarise({L: res[L][n] for L in T.LAYERS})
        summ[n] = s
        print(f"standalone {n:22s} sal {s['sal']:6.2f} churn {s['churn']:5.2f}", flush=True)
    if names:
        f = f"{H.HP}/standalone.json"
        allr = json.load(open(f)) if os.path.exists(f) else {}
        allr.update(summ)
        json.dump(allr, open(f, "w"), indent=1)
