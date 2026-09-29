"""train v2-recipe models on chain subsets of calib-fit (+ optional sm120tf tasks).
  train.py NAME --cf CHAINS --tf TASKS [--cap 1|4] [--stuck] [--val CHAINS] [--frac-rows F]
CHAINS: comma list of chain indices or 'none'; TASKS: comma list of sm120tf task names or 'none'."""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scalelib as S
R = f"{S.OUT}/private/rows"
F9 = S.V2F

def load(stream, chains, stuck, row_frac=1.0, seed=0):
    Xs, ys = [], []
    cols = list(range(9))
    if stuck:
        cols[2] = 9
    rng = np.random.default_rng(seed)
    for L in S.LAYERS:
        d = np.load(f"{R}/{stream}/L{L}.npz")
        m = np.isin(d["chain"], chains)
        if row_frac < 1:          # subsample blocks (groups of 101 rows)
            nb = len(m) // 101
            kb = rng.random(nb) < row_frac
            m &= np.repeat(kb, 101)
        Xs.append(d["X"][m][:, cols]); ys.append(d["y"][m])
    return np.concatenate(Xs), np.concatenate(ys)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name"); ap.add_argument("--cf", default="none"); ap.add_argument("--tf", default="none")
    ap.add_argument("--val", default="3,7,11,15,19,23,27,31"); ap.add_argument("--cap", type=int, default=1)
    ap.add_argument("--stuck", action="store_true"); ap.add_argument("--rowfrac", type=float, default=1.0)
    ap.add_argument("--tfw", type=float, default=1.0, help="sample weight of sm120tf rows"); ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--obj", default="tweedie:1.5")
    a = ap.parse_args()
    import lightgbm as lgb
    t0 = time.time()
    Xs, ys, ws = [], [], []
    if a.cf != "none":
        X, y = load("calib-fit", [int(c) for c in a.cf.split(",")], a.stuck, a.rowfrac); Xs.append(X); ys.append(y); ws.append(np.ones(len(y), np.float32))
    if a.tf != "none":
        names = json.load(open(f"{S.BLK}/sm120tf/meta.json"))["chains"]
        names = [n for n, s, e in zip(names, json.load(open(f"{S.BLK}/sm120tf/meta.json"))["bstart"][:-1], json.load(open(f"{S.BLK}/sm120tf/meta.json"))["bstart"][1:]) if e > s]
        idx = [names.index(t) for t in a.tf.split(",")]
        X, y = load("sm120tf", idx, a.stuck); Xs.append(X); ys.append(y); ws.append(np.full(len(y), a.tfw, np.float32))
    X, y, w = np.concatenate(Xs), np.concatenate(ys), np.concatenate(ws)
    del Xs, ys
    Xv, yv = load("calib-fit", [int(c) for c in a.val.split(",")], a.stuck)
    obj, _, pw = a.obj.partition(":")
    p = dict(objective=obj, metric=obj, learning_rate=0.3, num_leaves=15, min_data_in_leaf=500, bagging_fraction=0.5,
             bagging_freq=1, bagging_seed=3, feature_fraction=0.9, seed=0, num_threads=a.threads, verbosity=-1, max_bin=255)
    if obj == "tweedie":
        p["tweedie_variance_power"] = float(pw or 1.5)
    iters = 60
    if a.cap == 4:
        p["num_leaves"] = 31; iters = 120
    elif a.cap == 16:
        p["num_leaves"] = 63; iters = 240; p["learning_rate"] = 0.15
    dt = lgb.Dataset(X, y, weight=w if a.tf != "none" else None, feature_name=F9, free_raw_data=True)
    dv = lgb.Dataset(Xv, yv, reference=dt)
    ev = {}
    bst = lgb.train(p, dt, num_boost_round=iters, valid_sets=[dv], valid_names=["val"],
                    callbacks=[lgb.early_stopping(10, verbose=False), lgb.record_evaluation(ev)])
    bi = bst.best_iteration or iters
    out = f"{S.OUT}/models/{a.name}.txt"
    bst.save_model(out, num_iteration=bi)
    json.dump(dict(vars(a), n_train=int(len(y)), best_iter=bi, val_curve=ev["val"][obj], params=p, wall=time.time() - t0),
              open(out + ".meta.json", "w"), indent=1)
    print(a.name, "rows", len(y), "best", bi, "val", ev["val"][obj][bi - 1], f"{time.time() - t0:.0f}s", flush=True)

main()
