#!/usr/bin/env python3
"""T32 offline sim (= Adapt._core_gbdt next_refresh + hysteresis, parity-tested in parity.py) on a traced corpus:
share of the corpus' non-fixed salience (sum w^2|x|^2) and routed hits that the floating 51 serve at level 4.
  sim.py CORPUS NAME=MODEL [NAME=MODEL ...] [--oracles]   -> json on stdout + $OUT/sim_CORPUS.json"""
import json
import os
import sys
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402

import t32lib as T  # noqa: E402

corpus = sys.argv[1]
models = dict(x.split("=", 1) for x in sys.argv[2:] if "=" in x)
oracles = "--oracles" in sys.argv
fixed, fdef = T.serve_sets()


def oracle_serve(M, fx, nbc):
    """top-51 non-fixed by actual M summed over blocks [k, k + H/G) within the chain (T18 _core_oracle)."""
    nb = M.shape[0]
    out = np.zeros(M.shape, bool)
    for c0 in range(0, nb, nbc):
        s = M[c0:c0 + nbc]
        cs = np.vstack([np.zeros((1, 256)), np.cumsum(s, 0)])
        k = np.arange(s.shape[0])
        sc = cs[np.minimum(k + T.H // T.G, s.shape[0])] - cs[k]
        sc[:, fx] = -np.inf
        top = np.argsort(-sc, 1, kind="stable")[:, :51]
        np.put_along_axis(out[c0:c0 + nbc], top, True, 1)
    return out


def job(L):
    import lightgbm as lgb
    band = os.environ.get("T32_BAND", "")
    d = np.load(f"{T.OUT}/rows{'_band' + band if band else ''}/{corpus}/L{L}.npz")
    fx = np.zeros(256, bool); fx[fixed[L]] = True
    bc, bs = d["bcnt"].astype(np.float64), d["bsal"].astype(np.float64)
    nf = ~fx
    tot_c, tot_s = (bc * nf).sum(), (bs * nf).sum()
    out = {}
    X = d["X"].reshape(-1, 5)
    sets = {}
    X2 = None
    for name, path in models.items():
        scale = path.endswith("@mps")            # gbdt x EMA128 salience/hit (D3 gbdt_x_sal analogue, no delta)
        path = path.removesuffix("@mps")
        b = lgb.Booster(model_file=path)
        if scale and X2 is None:
            X2 = np.load(f"{T.OUT}/rows_v2{'_band' + band if band else ''}/{corpus}/L{L}.npz")["X2"]
        names = b.feature_name()
        Xm = X if tuple(names) == T.FEATS5 else T.feature_matrix(names, corpus, L, band=band, d=d)
        pred = b.predict(Xm, num_threads=1)
        if scale:
            pred = pred * X2[..., 3].ravel()
        S = T.score_blocks(pred, d["cand"], d["top"], d["e256"])
        sets[name] = T.sim_layer(S, fixed[L], fdef[L], hm=float(os.environ.get("T32_HM", "0.5")),
                                   lag=int(os.environ.get("T32_LAG", "1")))
    if oracles:
        nbc = T.CHAIN * T.SEQ // T.G
        sets["orc_count"] = oracle_serve(bc, fx, nbc)
        sets["orc_sal"] = oracle_serve(bs, fx, nbc)
    extra = {}
    if os.environ.get("T32_EXTRA"):                  # long-horizon analysis (build_lh.py rows; pooled num/den)
        lh = np.load(f"{T.OUT}/rows_lh/{corpus}/L{L}.npz")
        nbc = T.CHAIN * T.SEQ // T.G
        q = np.zeros_like(lh["qp"]); q[1:] = lh["qp"][:-1]; q[::nbc] = False    # decided at end of k-1, serves k
        q &= nf
        pos = np.arange(bs.shape[0]) % nbc
        R = {G: lh[f"ret{G}_sal"].astype(np.float64) for G in (256, 1024)}
        Rc = {G: lh[f"ret{G}_cnt"].astype(np.float64) for G in (256, 1024)}
        extra["_den"] = dict(qp=float((bs * q).sum()), nonfixed=float(tot_s), all=float(bs.sum()),
                             **{f"ret{G}": float(R[G].sum()) for G in R}, **{f"ret{G}_n": float(Rc[G].sum()) for G in R},
                             **{f"pos{a}": float((bs * nf)[(pos >= a) & (pos < b)].sum())
                                for a, b in ((0, 128), (128, 256), (256, 512))})
    for name, sv in sets.items():
        out[name] = dict(sal=float((bs * sv).sum() / tot_s), cnt=float((bc * sv).sum() / tot_c),
                         churn=float((sv[1:] & ~sv[:-1]).sum(1).mean()))
        if extra:
            hot = sv | fx
            out[name]["_num"] = dict(qp=float((bs * q * sv).sum()), nonfixed=float((bs * sv).sum()),
                                     **{f"ret{G}": float((R[G] * hot).sum()) for G in R},
                                     **{f"ret{G}_n": float((Rc[G] * hot).sum()) for G in R},
                                     **{f"pos{a}": float((bs * sv)[(pos >= a) & (pos < b)].sum())
                                        for a, b in ((0, 128), (128, 256), (256, 512))})
    if extra:
        out["_den"] = extra["_den"]
    return L, out


if __name__ == "__main__":
    with Pool(int(os.environ.get("NPROC", "38"))) as p:
        res = dict(p.map(job, T.LAYERS))
    names = [n for n in res[T.LAYERS[0]] if n != "_den"]
    summ = {n: {k: float(np.mean([res[L][n][k] for L in T.LAYERS])) for k in ("sal", "cnt", "churn")} for n in names}
    if "_den" in res[T.LAYERS[0]]:                 # pooled over layers: qp / pos* = floating coverage of non-fixed
        den = {k: sum(res[L]["_den"][k] for L in T.LAYERS) for k in res[T.LAYERS[0]]["_den"]}   # ret* = all-slot hot
        summ["_share"] = {"qp_of_nonfixed": den["qp"] / den["nonfixed"], "ret256_of_all": den["ret256"] / den["all"],
                          "ret1024_of_all": den["ret1024"] / den["all"], "ret256_n": den["ret256_n"],
                          "ret1024_n": den["ret1024_n"]}
        for n in names:
            num = {k: sum(res[L][n]["_num"][k] for L in T.LAYERS) for k in res[T.LAYERS[0]][n]["_num"]}
            summ[n].update({f"x_{k}": num[k] / max(den[k], 1e-30) for k in num})
    for n in names:
        print(f"{n:20s} nonfixed-sal covered {summ[n]['sal']:.4f}  hits covered {summ[n]['cnt']:.4f}  "
              f"churn/refresh {summ[n]['churn']:.2f}  " +
              " ".join(f"{k[2:]} {v:.4f}" for k, v in summ[n].items() if k.startswith("x_")))
    if "_share" in summ:
        print("shares", {k: round(v, 5) for k, v in summ["_share"].items()})
    json.dump({"corpus": corpus, "models": models, "summary": summ, "per_layer": res},
              open(f"{T.OUT}/sim_{corpus}{os.environ.get('T32_TAG', '')}.json", "w"), indent=1)
