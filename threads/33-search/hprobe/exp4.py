#!/usr/bin/env python3
"""exp4: residual probes (exp3 arithmetic) trained on calib-fit + hpx (github + c2048x + non-tb traces; NO nq-tail,
wikitext, vllm-docs), evaluated on glm52-heldout (fit on all train) and calib-val (fit excl. calib chains%4==3).
  exp4.py ARMS LAM TRAINSETS [NPROC]    TRAINSETS: cfhpx,hpx,cf"""
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
from multiprocessing import Pool  # noqa: E402
import numpy as np  # noqa: E402
import heval as H  # noqa: E402
import hplib as HL  # noqa: E402
import t32lib as T  # noqa: E402

P = f"{H.HP}/private"
PCAK = int(os.environ.get("PCAK", "512"))
HMS = (0.3, 0.4, 0.5)
ALPHAS = (0.5, 1.0)
CS = ("calib-fit", "glm52-heldout", "hpx")
SAVE = os.environ.get("SAVE", "")                    # "arm|train|lam|a" -> scores_glm52-heldout


def rows(c, L):
    return np.load(f"{P}/rows_hpx/L{L}.npz") if c == "hpx" else H.rows(c, L)


def pool(c, L):
    return HL.load_pool(L, c, keys=("h", "lg"), d=f"{P}/pool_hpx" if c == "hpx" else HL.POOL)


def ridge(X, Y, lam):
    ok = np.isfinite(Y).all(1)
    X, Y = X[ok], Y[ok]
    mx, sx = X.mean(0), X.std(0) + 1e-6
    Z = np.hstack([(X - mx) / sx, np.ones((len(X), 1))]).astype(np.float64)
    A = Z.T @ Z; A[np.diag_indices(A.shape[0] - 1)] += lam * len(Z) * 1e-3
    W = np.linalg.solve(A, Z.T @ Y)
    return lambda Xq: (np.hstack([(Xq - mx) / sx, np.ones((len(Xq), 1))]) @ W).astype(np.float32)


def pca(x, L):
    mu = x.mean(0)
    xs = (x[::2] - mu).astype(np.float32)
    rng = np.random.default_rng(L)
    Q = np.linalg.qr(xs.T @ (xs @ rng.standard_normal((xs.shape[1], PCAK + 64)).astype(np.float32)))[0]
    Q = np.linalg.qr(xs.T @ (xs @ Q))[0]
    _, _, Vt = np.linalg.svd(xs @ Q, full_matrices=False)
    return mu, (Q @ Vt.T)[:, :PCAK].astype(np.float32)


def job(args):
    L, arms, lam, trains = args
    CS = ("calib-fit", "glm52-heldout") + (("hpx",) if any("hpx" in t for t in trains) else ())
    d = {c: rows(c, L) for c in CS}
    off = {c: np.log1p(np.maximum(np.load(f"{P}/v2S_{c}/L{L}.npy"), 0)) for c in CS}
    Y = {c: np.log1p(HL.target(d[c], L)) - off[c] for c in CS if c != "glm52-heldout"}
    raw = {c: pool(c, L) for c in CS}
    lg = {c: np.hstack([raw[c][1], HL.bmean(raw[c][1], 4), HL.bema(raw[c][1], 256)]).astype(np.float32) for c in CS}
    h16 = {c: raw[c][0].astype(np.float32) for c in CS}
    h64 = {c: HL.bmean(h16[c], 4) for c in CS}
    del raw
    val = HL.chain_id(len(Y["calib-fit"])) % 4 == 3
    out = {}
    for tr in trains:
        for split in ("all", "cv"):
            sel = []                                        # (corpus, row mask)
            if tr in ("cf", "cfhpx"):
                sel.append(("calib-fit", ~val if split == "cv" else np.ones(len(val), bool)))
            if tr in ("hpx", "cfhpx"):
                sel.append(("hpx", np.ones(len(Y["hpx"]), bool)))
            mu, V = pca(np.vstack([h64[c][m] for c, m in sel]), L)
            hf = {c: np.hstack([(h16[c] - mu) @ V, (h64[c] - mu) @ V]) for c in CS}
            feats = {"lg": lg, "h": hf, "lgh": {c: np.hstack([lg[c], hf[c]]) for c in CS}}
            ev = "glm52-heldout" if split == "all" else "calib-fit"
            de = d[ev]; bs, bc = de["bsal"].astype(np.float64), de["bcnt"].astype(np.float64)
            for arm in arms:
                f = ridge(np.vstack([feats[arm][c][m] for c, m in sel]), np.vstack([Y[c][m] for c, m in sel]), lam)
                Pe = f(feats[arm][ev])
                for a in ALPHAS:
                    S = np.expm1(off[ev] + a * Pe).astype(np.float32)
                    k = f"{arm}|{tr}|{lam:g}|{a:g}"
                    if split == "all" and SAVE == k:
                        os.makedirs(f"{H.HP}/scores_glm52-heldout", exist_ok=True)
                        np.save(f"{H.HP}/scores_glm52-heldout/L{L}.npy", S.astype(np.float16))
                    for hm in HMS:
                        out[f"{k}|{split}|hm{hm}"] = H.eval_S(S, L, bs, bc, hm=hm, mask=val if split == "cv" else None)
    for split, ev in (("all", "glm52-heldout"), ("cv", "calib-fit")):
        S = np.expm1(off[ev]).astype(np.float32); de = d[ev]
        for hm in HMS:
            out[f"v2|{split}|hm{hm}"] = H.eval_S(S, L, de["bsal"].astype(np.float64), de["bcnt"].astype(np.float64),
                                                 hm=hm, mask=val if split == "cv" else None)
    return L, out


if __name__ == "__main__":
    arms = sys.argv[1].split(","); lam = float(sys.argv[2]); trains = sys.argv[3].split(",")
    nproc = int(sys.argv[4]) if len(sys.argv) > 4 else 6
    Ls = [int(x) for x in os.environ["LAYERS"].split(",")] if os.environ.get("LAYERS") else T.LAYERS
    with Pool(nproc) as p:
        res = dict(p.imap_unordered(job, [(L, arms, lam, trains) for L in Ls]))
    summ = {}
    for n in sorted(res[Ls[0]]):
        s = H.summarise({L: res[L][n] for L in Ls}); summ[n] = s
        print(f"{n:36s} sal {s['sal']:6.2f} churn {s['churn']:5.2f}", flush=True)
    json.dump(summ, open(f"{H.HP}/exp4_{os.environ.get('TAGX', 'all')}.json", "w"), indent=1)
