"""serve cost: ms per refresh = LightGBM predict on 75 layers x NROWS candidates (band all: 256), threads 1 / 8;
model file size.  g33time.py MODEL.txt ..."""
import os, sys, time
import numpy as np
import lightgbm as lgb
NR = int(os.environ.get("NROWS", "256"))
for path in sys.argv[1:]:
    b = lgb.Booster(model_file=path)
    nf = b.num_feature()
    X = np.random.default_rng(0).gamma(0.5, 1.0, (75 * NR, nf)).astype(np.float32)
    r = {}
    for th in (1, 8):
        b.predict(X[:100], num_threads=th)
        ts = []
        for _ in range(20):
            t = time.perf_counter(); b.predict(X, num_threads=th); ts.append(time.perf_counter() - t)
        r[th] = 1e3 * np.median(ts)
    print(f"{os.path.basename(path):20s} trees {b.num_trees():4d} feats {nf:2d} size {os.path.getsize(path) / 1e3:7.1f} kB  "
          f"ms/refresh (75x{NR} rows) 1thr {r[1]:6.2f}  8thr {r[8]:6.2f}", flush=True)
