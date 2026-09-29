"""train.py NAME FEATS [iters] [obj]   FEATS = comma list; 'v2' = the 9 v2 feats, 'hk'/'hmm'/'bo'/'kf' = families."""
import os, sys, json, time
import numpy as np
import plib as P, feats as FE, rows as R
import lightgbm as lgb
name, spec = sys.argv[1], sys.argv[2]
iters = int(sys.argv[3]) if len(sys.argv) > 3 else 60
fl = []
for t in spec.split(","):
    fl += P.V2F if t == "v2" else FE.FAM.get(t, [t])
ci = [R.COLS.index(f) for f in fl]
t0 = time.time()
Xt, yt, Xv, yv = [], [], [], []
for L in P.T.LAYERS:
    d = np.load(f"{R.RD}/L{L}.npz")
    Xt.append(d["Xtr"][:, ci]); yt.append(d["ytr"]); Xv.append(d["Xva"][:, ci]); yv.append(d["yva"])
Xt, yt, Xv, yv = map(np.concatenate, (Xt, yt, Xv, yv))
print(name, fl, "rows", len(yt), len(yv), f"{time.time()-t0:.0f}s", flush=True)
p = dict(objective="tweedie", metric="tweedie", tweedie_variance_power=1.5, learning_rate=0.3, num_leaves=15,
         min_data_in_leaf=500, bagging_fraction=0.5, bagging_freq=1, bagging_seed=3, feature_fraction=0.9, seed=0,
         num_threads=int(os.environ.get("THR", "20")), verbosity=-1, max_bin=255)
dt = lgb.Dataset(Xt, yt, feature_name=fl, free_raw_data=True)
dv = lgb.Dataset(Xv, yv, reference=dt)
ev = {}
b = lgb.train(p, dt, num_boost_round=iters, valid_sets=[dv], valid_names=["val"],
              callbacks=[lgb.early_stopping(10, verbose=False), lgb.record_evaluation(ev)])
bi = b.best_iteration or iters
os.makedirs(f"{P.OUT}/models", exist_ok=True)
b.save_model(f"{P.OUT}/models/{name}.txt", num_iteration=bi)
imp = dict(zip(fl, b.feature_importance("gain", iteration=bi).round(0).tolist()))
json.dump(dict(feats=fl, best_iter=bi, val=ev["val"]["tweedie"][bi - 1], imp=imp, params=p), open(f"{P.OUT}/models/{name}.json", "w"), indent=1)
print(name, "best", bi, "val tweedie", round(ev["val"]["tweedie"][bi - 1], 5), "imp", imp, f"{time.time()-t0:.0f}s", flush=True)
