#!/usr/bin/env python3
"""T32 k=0 fixed-set arm (offline): no fixed experts, all 77 level-4 slots floating (v2 model, sync, band all =
every expert a candidate).  Builds rows_bandk0/CORPUS (fixed = {}) + rows_v2_bandk0, then sweeps the hysteresis
margin hm and reports all-slot sal-hot / routes-hot / churn (new floating experts per 16-token refresh) next to the
k=26 reference (v2 sync band all, hm 0.5 = 74.77 / 2.78 heldout).
  k0.py build CORPUS [NPROC]          k0.py sweep CORPUS [hm,hm,...]   -> $OUT/k0_sweep_CORPUS.json"""
import json
import os
import sys
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402

import t32lib as T  # noqa: E402

cmd, corpus = sys.argv[1], sys.argv[2]
V2 = "/home/coder/git/nestquant/streaming/gbdt_v2sal_p64.txt"
fixed, fdef = T.serve_sets()
NOFIX = {L: [] for L in fixed}
FDEF77 = {L: list(fixed[L]) + [e for e in fdef[L] if e not in set(fixed[L])] for L in fixed}
K = int(os.environ.get("K", "0"))  # K>0: fixed = top-K of fixed-26 by calib-fit salience, rest floating (nf = 77-K)
if K:
    import json as _j
    _m = _j.load(open(f"{T.OUT}/k{K}_manifest.json"))
    KFIX = {L: _m["default_allocation"][str(L)] for L in fixed}
    KFDEF = {L: _m["floating_default"][str(L)] for L in fixed}
RK, RV = f"{T.OUT}/rows_bandk0/{corpus}", f"{T.OUT}/rows_v2_bandk0/{corpus}"


def build(L):
    if not os.path.exists(f"{RK}/L{L}.npz"):
        T.build_layer(L, corpus, NOFIX, RK, 0, 256)
    d = np.load(f"{RK}/L{L}.npz")
    os.makedirs(RV, exist_ok=True)
    np.savez(f"{RV}/L{L}.npz", X2=T.v2_features(d["bcnt"], d["bsal"], d["cand"]))
    return L


def sweep(L):
    import lightgbm as lgb
    b = lgb.Booster(model_file=V2)
    d = np.load(f"{RK}/L{L}.npz")
    X = np.concatenate([d["X"], np.load(f"{RV}/L{L}.npz")["X2"]], -1).reshape(-1, 9)
    S = T.score_blocks(b.predict(X, num_threads=1), d["cand"], d["top"], d["e256"])
    bc, bs = d["bcnt"].astype(np.float64), d["bsal"].astype(np.float64)
    out = {}
    for hm in HMS:
        if K:
            sv = T.sim_layer(S, KFIX[L], KFDEF[L], nf=77 - K, hm=hm, lag=0)
            sv = sv.copy(); sv[:, KFIX[L]] = True
        else:
            sv = T.sim_layer(S, [], FDEF77[L], nf=77, hm=hm, lag=0)
        out[hm] = dict(sal=float((bs * sv).sum() / bs.sum()), cnt=float((bc * sv).sum() / bc.sum()),
                       churn=float((sv[1:] & ~sv[:-1]).sum(1).mean()))
    return L, out


if __name__ == "__main__":
    if cmd == "build":
        with Pool(int(sys.argv[3]) if len(sys.argv) > 3 else 16) as p:
            print(sorted(p.map(build, T.LAYERS))[-1])
    else:
        HMS = [float(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "0.5,1,1.5,2,3,4,6").split(",")]
        with Pool(int(os.environ.get("NPROC", "16"))) as p:
            res = dict(p.map(sweep, T.LAYERS))
        summ = {}
        for hm in HMS:
            s = {k: float(np.mean([res[L][hm][k] for L in T.LAYERS])) for k in ("sal", "cnt", "churn")}
            summ[str(hm)] = s
            print(f"[{corpus}] k={K} nf={77 - K} hm {hm:4.1f}  sal-hot {s['sal'] * 100:6.2f}  routes-hot {s['cnt'] * 100:6.2f}"
                  f"  churn {s['churn']:5.2f}", flush=True)
        json.dump(dict(summary=summ, per_layer={str(L): {str(h): v for h, v in res[L].items()} for L in T.LAYERS}),
                  open(f"{T.OUT}/k{K}_sweep_{corpus}.json", "w"), indent=1)
