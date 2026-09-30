"""step() timing, 75 layers, sync mode, 8 threads: v2 (9 feats) vs D0 (18 feats) at k0 (n_float 77), candidate band
rlo..rhi = 0..256 (all experts) and 0..160.  Synthetic routing (8 of 256 per token, Zipf-ish per layer), 1 token per
decode step; reports per-refresh (block-boundary step) and per-plain-step wall time.  time_d0.py MODEL_D0 MODEL_V2"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/coder/git/nestquant/streaming")
from gbdt_predictor_d0 import GBDTPredictorD0
from gbdt_predictor_v2 import GBDTPredictorV2

MD0, MV2 = sys.argv[1], sys.argv[2]
LAYERS = list(range(3, 78)); NL, NE = len(LAYERS), 256
rng = np.random.default_rng(0)
pr = rng.dirichlet(np.full(NE, 0.3), NL)
NBLK, WARM = 60, 20
toks = [np.stack([rng.choice(NE, 8, replace=False, p=pr[i]) for i in range(NL)]) for _ in range(NBLK * 16)]
for name, cls, mp in (("v2", GBDTPredictorV2, MV2), ("d0", GBDTPredictorD0, MD0)):
    for rhi in (256, 160):
        p = cls(LAYERS, {L: [] for L in LAYERS}, model_path=mp, n_float=77, hm=0.6, mode="sync", num_threads=8,
                rlo=0, rhi=rhi)
        tr, ts = [], []
        for t, ids in enumerate(toks):
            c = np.zeros((NL, NE), np.float32); np.put_along_axis(c, ids, 1.0, 1)
            sal = c * rng.gamma(2.0, 0.5, (NL, NE))
            t0 = time.perf_counter(); r = p.step(c, ntok=1, sal=sal); dt = time.perf_counter() - t0
            if t >= WARM * 16:
                (tr if r else ts).append(dt)
        tg = []
        want = p.target(np.zeros((NL, NE), bool))
        for _ in range(50):
            t0 = time.perf_counter(); want = p.target(want); tg.append(time.perf_counter() - t0)
        p.close()
        print(f"{name} rows/layer {rhi:3d}: refresh step median {np.median(tr)*1e3:.2f} ms p90 {np.percentile(tr, 90)*1e3:.2f} ms "
              f"(n {len(tr)}); plain step median {np.median(ts)*1e3:.3f} ms; target() {np.median(tg)*1e3:.2f} ms", flush=True)
