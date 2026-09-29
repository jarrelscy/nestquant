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
    d = np.load(f"{T.OUT}/rows/{corpus}/L{L}.npz")
    fx = np.zeros(256, bool); fx[fixed[L]] = True
    bc, bs = d["bcnt"].astype(np.float64), d["bsal"].astype(np.float64)
    nf = ~fx
    tot_c, tot_s = (bc * nf).sum(), (bs * nf).sum()
    out = {}
    X = d["X"].reshape(-1, 5)
    sets = {}
    for name, path in models.items():
        pred = lgb.Booster(model_file=path).predict(X, num_threads=1)
        S = T.score_blocks(pred, d["cand"], d["top"], d["e256"])
        sets[name] = T.sim_layer(S, fixed[L], fdef[L])
    if oracles:
        nbc = T.CHAIN * T.SEQ // T.G
        sets["orc_count"] = oracle_serve(bc, fx, nbc)
        sets["orc_sal"] = oracle_serve(bs, fx, nbc)
    for name, sv in sets.items():
        out[name] = dict(sal=float((bs * sv).sum() / tot_s), cnt=float((bc * sv).sum() / tot_c),
                         churn=float((sv[1:] & ~sv[:-1]).sum(1).mean()))
    return L, out


if __name__ == "__main__":
    with Pool(int(os.environ.get("NPROC", "38"))) as p:
        res = dict(p.map(job, T.LAYERS))
    names = list(res[T.LAYERS[0]])
    summ = {n: {k: float(np.mean([res[L][n][k] for L in T.LAYERS])) for k in ("sal", "cnt", "churn")} for n in names}
    for n in names:
        print(f"{n:20s} nonfixed-sal covered {summ[n]['sal']:.4f}  hits covered {summ[n]['cnt']:.4f}  "
              f"churn/refresh {summ[n]['churn']:.2f}")
    json.dump({"corpus": corpus, "models": models, "summary": summ, "per_layer": res},
              open(f"{T.OUT}/sim_{corpus}.json", "w"), indent=1)
