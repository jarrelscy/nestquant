"""train horizon / quantile heads on $W/train_s4.npz.  train_heads.py NAME TARGET_IDX OBJ [ALPHA] [ITERS]
  TARGET_IDX: 0 sal16, 1 sal64, 2 sal128, 3 sal256, 4 cnt64 ; OBJ tweedie|quantile|poisson|l2"""
import sys, time, json, numpy as np, lightgbm as lgb
import dlib as D
name, ti, obj = sys.argv[1], int(sys.argv[2]), sys.argv[3]
alpha = float(sys.argv[4]) if len(sys.argv) > 4 else 0.9
iters = int(sys.argv[5]) if len(sys.argv) > 5 else 60
t0 = time.time()
z = np.load(f"{D.W}/train_s4.npz"); fe = list(z["feats"])
Xt, yt, Xv, yv = z["Xt"], z["Yt"][:, ti], z["Xv"], z["Yv"][:, ti]
p = dict(objective=obj, learning_rate=0.3, num_leaves=15, min_data_in_leaf=500, bagging_fraction=0.5, bagging_freq=1,
         bagging_seed=3, feature_fraction=0.9, seed=0, num_threads=20, verbosity=-1, max_bin=255)
if obj == "tweedie": p["tweedie_variance_power"] = 1.5
if obj == "quantile": p["alpha"] = alpha; p["learning_rate"] = 0.15
dt = lgb.Dataset(Xt, yt, feature_name=fe, free_raw_data=True); dv = lgb.Dataset(Xv, yv, reference=dt)
ev = {}
b = lgb.train(p, dt, num_boost_round=iters, valid_sets=[dv], valid_names=["val"],
              callbacks=[lgb.early_stopping(10, verbose=False), lgb.record_evaluation(ev), lgb.log_evaluation(20)])
bi = b.best_iteration or iters
b.save_model(f"{D.W}/models/{name}.txt", num_iteration=bi)
json.dump(dict(target=ti, obj=obj, alpha=alpha, best=bi, wall=time.time() - t0), open(f"{D.W}/models/{name}.json", "w"))
print(name, "best", bi, f"{time.time()-t0:.0f}s", flush=True)
