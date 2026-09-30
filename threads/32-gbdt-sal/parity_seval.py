#!/usr/bin/env python3
"""T32 nf_map parity vs T33k seval (alloc ALLOC runs, sm120tf pooled stream): for layer L and allocation NAME, rebuild
seval's v2 scores (scalelib features, its model) and replay (a) T33k alib.sim_seg with seval's start set and (b) the
T32 offline replay t32lib.sim_layer (bitwise == harness Adapt, see parity_k) with kB_manifest floating_default and
nf_map[L], chain per seval segment; check sets identical, start sets identical, and the per-layer sal / churn equal
to seval_sm120tf_pool.json v2NAME|0|HM.   parity_seval.py L NAME HM"""
import json
import os
import sys
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np  # noqa: E402
sys.path.insert(0, "/home/coder/git/nestquant/threads/33-search/alloc")
sys.path.insert(0, "/home/coder/git/nestquant/threads/33-search/scale")
import alib as A  # noqa: E402
import scalelib as SL  # noqa: E402
import t32lib as T  # noqa: E402
import lightgbm as lgb  # noqa: E402

L, name, hm = int(sys.argv[1]), sys.argv[2], float(sys.argv[3])
V2 = "/tmp/nestquant/32-gbdt-sal/models/v2_sal_tweedie1.5.txt"      # seval's model
nf = json.load(open(f"{A.A}/alloc_{name}.json"))[str(L)]
man = json.load(open(f"{T.OUT}/kB_manifest.json"))["floating_default"][str(L)]
f26 = A.f26_ranked(L)
st = f26 + list(A.fdef[L]); cal = A.load("calib-fit", L)[1].sum(0)
st = st + [int(e) for e in np.argsort(-cal, kind="stable") if e not in set(st)]
D = SL.load("sm120tf", L); sg = D["sg"]
F, e256 = SL.feats(D)
nb = F.shape[0]
S = lgb.Booster(model_file=V2).predict(F.reshape(-1, 9), num_threads=4).reshape(nb, 256).astype(np.float32)
sa = A.sim_seg(S, [], st, nf, hm, sg=sg)
sb = np.concatenate([T.sim_layer(S[s:e], [], man, nf=nf, hm=hm, nbc=e - s, lag=0) for s, e in sg])
bs = D["bs"]
ref = json.load(open(f"{A.A}/seval_sm120tf_pool.json"))["per_layer"][str(L)][f"v2{name}|0|{hm}"]
sal = float((bs * sb).sum() / bs.sum()); ch = A.churn_seg(sb, sg)
print(f"L{L} {name} nf={nf} hm={hm}: start-set equal {st[:nf] == man[:nf]}  sets equal {bool((sa == sb).all())} "
      f"({int((sa != sb).any(1).sum())}/{nb} blocks differ)  served size {int(sb.sum(1).min())}-{int(sb.sum(1).max())}  "
      f"sal {sal:.6f} vs seval {ref['sal']:.6f}  churn {ch:.4f} vs {ref['churn']:.4f}", flush=True)
