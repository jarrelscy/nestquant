#!/usr/bin/env python3
"""exp3: residual probes on top of v2.  Target R = log1p(next-64 normalised salience) - log1p(v2 score) per
(block, expert); ridge (OOF by chain%4 on calib-fit; full fit for heldout) from pooled inputs of the own layer:
  lg  = router logits of the pooled context (last 16, last 64, EMA256)          768
  h   = PCA-K of the RMS-normalised residual h_mid pooled over last 16 / last 64  2K
Scores: S = expm1(log1p(v2) + a * P).  Saves P as feature rp_{arm}_l{lam} for exp2 stacking.
  exp3.py LAMS ARMS [NPROC]      e.g. exp3.py 100,1000,10000 lg,h,lgh 8"""
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
from multiprocessing import Pool  # noqa: E402
import numpy as np  # noqa: E402
import heval as H  # noqa: E402
import hplib as HL  # noqa: E402
import make_feats as M  # noqa: E402
import t32lib as T  # noqa: E402

CORP = M.CORP
PCAK = M.PCAK
ALPHAS = (0.5, 1.0)


def inputs(L, need):
    ins = {}
    if "lg" in need:
        ins["lg"] = {}
        for c in CORP:
            lg = M.get_lg(L, c)
            ins["lg"][c] = np.hstack([lg, HL.bmean(lg, 4), HL.bema(lg, 256)])
    if "h" in need:
        hs = {c: M.get_h(L, c) for c in CORP}
        x = HL.bmean(hs["calib-fit"], 4)
        mu = x.mean(0)
        xs = (x[::2] - mu).astype(np.float32)
        rng = np.random.default_rng(L)
        Q = np.linalg.qr(xs.T @ (xs @ rng.standard_normal((xs.shape[1], PCAK + 64)).astype(np.float32)))[0]
        Q = np.linalg.qr(xs.T @ (xs @ Q))[0]
        _, _, Vt = np.linalg.svd(xs @ Q, full_matrices=False)
        V = (Q @ Vt.T)[:, :PCAK].astype(np.float32)
        ins["h"] = {c: np.hstack([(hs[c] - mu) @ V, (HL.bmean(hs[c], 4) - mu) @ V]) for c in CORP}
        os.makedirs(f"{H.HP}/private/pca", exist_ok=True)
        np.savez(f"{H.HP}/private/pca/L{L}.npz", mu=mu, V=V)
    return ins


def job(args):
    L, lams, arms = args
    d = {c: H.rows(c, L) for c in CORP}
    off = {c: np.log1p(np.maximum(np.load(f"{H.HP}/private/v2S_{c}/L{L}.npy"), 0)) for c in CORP}
    Y = np.log1p(HL.target(d["calib-fit"], L)) - off["calib-fit"]
    need = set("".join(arms).replace("lg", "L").replace("h", "H"))
    ins = inputs(L, {"lg" if "L" in need else "", "h" if "H" in need else ""})
    ins["c"] = {c: np.zeros((len(d[c]["valid"]), 1), np.float32) for c in CORP}     # intercept-only control
    parts = {"lg": ["lg"], "h": ["h"], "lgh": ["lg", "h"], "c": ["c"]}
    out = {}
    dh, dc = d["glm52-heldout"], d["calib-fit"]
    val = HL.chain_id(len(Y)) % 4 == 3
    for arm in arms:
        Xc = np.hstack([ins[p]["calib-fit"] for p in parts[arm]]); Xh = np.hstack([ins[p]["glm52-heldout"] for p in parts[arm]])
        for lam in lams:
            M.LAM = lam
            Pc, Ph = M.fit_oof(Xc, Y, Xh)
            n = f"rp_{arm}_l{lam:g}"
            M.save(n, "calib-fit", L, Pc); M.save(n, "glm52-heldout", L, Ph)
            for a in ALPHAS:
                Sh = np.expm1(off["glm52-heldout"] + a * Ph).astype(np.float32)
                Sc = np.expm1(off["calib-fit"] + a * Pc).astype(np.float32)
                out[f"{n}_a{a:g}"] = H.eval_S(Sh, L, dh["bsal"].astype(np.float64), dh["bcnt"].astype(np.float64))
                out[f"{n}_a{a:g}|cv"] = H.eval_S(Sc, L, dc["bsal"].astype(np.float64), dc["bcnt"].astype(np.float64),
                                                 mask=val)
    Sc = np.expm1(off["calib-fit"]).astype(np.float32)
    out["v2|cv"] = H.eval_S(Sc, L, dc["bsal"].astype(np.float64), dc["bcnt"].astype(np.float64), mask=val)
    return L, out


if __name__ == "__main__":
    lams = [float(x) for x in sys.argv[1].split(",")]
    arms = sys.argv[2].split(",")
    nproc = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    Ls = [int(x) for x in os.environ["LAYERS"].split(",")] if os.environ.get("LAYERS") else T.LAYERS
    with Pool(nproc) as p:
        res = dict(p.map(job, [(L, lams, arms) for L in Ls]))
    summ = {}
    for n in sorted(res[Ls[0]]):
        s = H.summarise({L: res[L][n] for L in Ls})
        summ[n] = s
        print(f"{n:28s} sal {s['sal']:6.2f} churn {s['churn']:5.2f}", flush=True)
    json.dump(summ, open(f"{H.HP}/exp3_{os.environ.get('TAGX', 'all')}.json", "w"), indent=1)
