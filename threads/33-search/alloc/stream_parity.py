#!/usr/bin/env python3
"""T33k alloc: offline-vs-streaming parity for the k0 arm (config-only serve change).
Streaming = streaming/gbdt_predictor_v2.GBDTPredictorV2 with the v2 model, fixed = {} (k0), n_float = 77,
band all (rlo=0, rhi=256), hm = HM, mode sync; fed token by token (counts, emitted token id, per-step salience), one
predictor per 4x2048-token chain (request start).  Offline = cached v2 S (cache_scores.py) + alib.sim_seg.
Checks: max |S_online - S_offline|, serve sets identical, sal-hot identical.   stream_parity.py [CORPUS] [HM]"""
import sys, time
import numpy as np
sys.path.insert(0, "/home/coder/git/nestquant/streaming")
import alib
from alib import T
from gbdt_predictor_v2 import GBDTPredictorV2

corpus = sys.argv[1] if len(sys.argv) > 1 else "glm52-heldout"
HM = float(sys.argv[2]) if len(sys.argv) > 2 else 0.7
MODEL = "/tmp/nestquant/32-gbdt-sal/models/v2_sal_tweedie1.5.txt"
LAY = T.LAYERS; NL = len(LAY); NF = 77
t0 = time.time()
tr = [T.load_layer(L, corpus) for L in LAY]
ids = np.stack([a[0] for a in tr], 1).astype(np.int64)      # [T, NL, 8]
w = np.stack([a[1] for a in tr], 1).astype(np.float64)
xn = np.stack([a[2] for a in tr], 1).astype(np.float64)     # [T, NL]
Tn = ids.shape[0]; nb = Tn // T.G
tok = T.tokens(corpus, Tn // T.SEQ)
off = [alib.load(corpus, L) for L in LAY]
print(f"loaded {Tn} tokens x {NL} layers in {time.time() - t0:.0f}s", flush=True)
# offline reference
ref = []
for i, L in enumerate(LAY):
    start = alib.f26_ranked(L) + alib.fdef[L]
    ref.append(alib.sim_seg(off[i][0], [], start, NF, HM))
# streaming
lofs = np.arange(NL)[:, None] * 256
serve = np.zeros((nb, NL, 256), bool); mx = 0.0; nref = 0
NBT = alib.NBC * T.G
for c0 in range(0, Tn, NBT):
    p = GBDTPredictorV2(LAY, {L: [] for L in LAY}, MODEL, n_float=NF, hm=HM, rlo=0, rhi=256, mode="sync",
                        num_threads=8)
    want = np.zeros((NL, 256), bool)
    for i, L in enumerate(LAY):
        want[i, (alib.f26_ranked(L) + alib.fdef[L])[:NF]] = True       # floating_default (77)
    for t in range(c0, min(c0 + NBT, Tn)):
        idx = (ids[t] + lofs).ravel()
        cnt = np.bincount(idx, minlength=NL * 256).reshape(NL, 256)
        sal = np.bincount(idx, weights=(w[t] ** 2 * xn[t][:, None]).ravel(), minlength=NL * 256).reshape(NL, 256)
        nt = [int(tok[t + 1])] if t + 1 < Tn else None
        if t % T.G == 0:
            serve[t // T.G] = want                                      # set in force during block t//G
        if p.step(cnt, 1, nt, new_request=(t == c0), sal=sal):
            b = t // T.G; nref += 1
            mx = max(mx, float(np.abs(p.S - np.stack([o[0][b] for o in off])).max()))
            want = p.target(want)
    p.close()
print(f"stream pass {time.time() - t0:.0f}s, {nref} refreshes; max |S_online - S_offline| = {mx:.3g}", flush=True)
mism = sum(int((serve[:, i] != ref[i]).any(1).sum()) for i in range(NL))
def salhot(sv):
    return 100 * np.mean([(off[i][1] * sv[:, i]).sum() / off[i][1].sum() for i in range(NL)])
print(f"serve-set mismatch blocks (sum over layers): {mism}", flush=True)
print(f"sal-hot online {salhot(serve):.4f}  offline {salhot(np.stack(ref, 1)):.4f}  churn online "
      f"{np.mean([alib.churn_seg(serve[:, i]) for i in range(NL)]):.3f}", flush=True)
