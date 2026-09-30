#!/usr/bin/env python3
"""predictor ms per refresh, 75 layers, 8 threads: streaming GBDTPredictorV2 (v2, k0, band = top-N by EMA256,
rlo=0 rhi=N) step() at refresh on heldout decode tokens (features + predict), and predict-only for v2 and T33g dyn0
(18 features; X = the v2 streaming features of the same refresh, extra 9 columns filled from related v2 columns --
tree cost depends on paths only, value realism approximate).   ptime.py [NREF]"""
import sys, time
import numpy as np
sys.path.insert(0, "/home/coder/git/nestquant/streaming")
import alib
from alib import T
from gbdt_predictor_v2 import GBDTPredictorV2
import lightgbm as lgb

NREF = int(sys.argv[1]) if len(sys.argv) > 1 else 150
V2 = "/tmp/nestquant/32-gbdt-sal/models/v2_sal_tweedie1.5.txt"; DYN = "/tmp/nestquant/33-search/gbdt/models/dyn0.txt"
LAY = T.LAYERS; NL = len(LAY); corpus = "glm52-heldout"
tr = [T.load_layer(L, corpus) for L in LAY]
n = (NREF + 64) * 16
ids = np.stack([a[0][:n] for a in tr], 1).astype(np.int64); w = np.stack([a[1][:n] for a in tr], 1).astype(np.float64)
xn = np.stack([a[2][:n] for a in tr], 1).astype(np.float64); tok = T.tokens(corpus, n // T.SEQ + 2)
lofs = np.arange(NL)[:, None] * 256
bd = lgb.Booster(model_file=DYN)
for N in (256, 192, 160, 128, 96):
    p = GBDTPredictorV2(LAY, {L: [] for L in LAY}, V2, n_float=77, hm=0.7, rlo=0, rhi=N, mode="sync", num_threads=8)
    ts, tf, tp, td = [], [], [], []
    for t in range(n):
        idx = (ids[t] + lofs).ravel()
        cnt = np.bincount(idx, minlength=NL * 256).reshape(NL, 256)
        sal = np.bincount(idx, weights=(w[t] ** 2 * xn[t][:, None]).ravel(), minlength=NL * 256).reshape(NL, 256)
        t0 = time.perf_counter()
        r = p.step(cnt, 1, [int(tok[t + 1])], new_request=(t == 0), sal=sal)
        if r:
            ts.append(time.perf_counter() - t0)
            if len(ts) > 64:                                 # warm state; time the pieces separately
                t1 = time.perf_counter(); f = p._features(); tf.append(time.perf_counter() - t1)
                X = f[0]
                t1 = time.perf_counter(); p.bst.predict(X, num_threads=8); tp.append(time.perf_counter() - t1)
                Xd = np.concatenate([X, X[:, [1, 6, 0, 5, 1, 6, 6, 1, 7]]], 1)
                t1 = time.perf_counter(); bd.predict(Xd, num_threads=8); td.append(time.perf_counter() - t1)
        if len(tf) >= NREF: break
    p.close()
    m = lambda a: 1e3 * float(np.median(a))  # noqa: E731
    print(f"N {N:3d} rows {NL * N:6d}: v2 step@refresh {m(ts[64:]):6.2f} ms (features {m(tf):5.2f} + predict {m(tp):5.2f});"
          f"  dyn0 predict-only {m(td):6.2f} ms", flush=True)
