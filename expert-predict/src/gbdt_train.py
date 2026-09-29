import sys, os, time, pickle; sys.path.insert(0, '/data/Jarrel/expert-predict/src'); from gbdt_feats import *
import lightgbm as lgb
TRAIN = sys.argv[1].split(','); VALT = sys.argv[2].split(','); tag = sys.argv[3]; EVERY = int(os.environ.get('EVERY', 64))
t0 = time.time(); Ps = coact_tables(TRAIN); np.save(f'{W}/feat/coact_{tag}.npy', Ps); print('coact', time.time() - t0, flush=True)
def build(tasks, every):
    Xs, y1, y2 = [], [], []
    for t in tasks:
        d = load(t)
        for b, cand, X, a, bb, *_ in gen(d, Ps, every): Xs.append(X.astype(np.float32)); y1.append(a); y2.append(bb)
        print(t, len(Xs), time.time() - t0, flush=True)
    return np.concatenate(Xs), np.concatenate(y1), np.concatenate(y2)
Xtr, ytr64, ytr256 = build(TRAIN, EVERY); Xva, yva64, yva256 = build(VALT, EVERY)
np.savez(f'{W}/feat/gbdt_rows_{tag}.npz', Xtr=Xtr, ytr64=ytr64, ytr256=ytr256, Xva=Xva, yva64=yva64, yva256=yva256)
print('rows', Xtr.shape, Xva.shape, flush=True)
for tgt in sys.argv[4].split(','):
    yt, yv, obj = {'p64': (ytr64, yva64, 'poisson'), 'p256': (ytr256, yva256, 'poisson'), 'b64': (ytr64 > 0, yva64 > 0, 'binary')}[tgt]
    dtr = lgb.Dataset(Xtr, yt.astype(np.float32), feature_name=FNAMES, free_raw_data=False)
    dva = lgb.Dataset(Xva, yv.astype(np.float32), reference=dtr)
    prm = dict(objective=obj, learning_rate=float(os.environ.get('LR', 0.1)), num_leaves=int(os.environ.get('LEAVES', 63)), min_data_in_leaf=500, feature_fraction=0.9, bagging_fraction=0.5,
               bagging_freq=1, num_threads=16, verbose=-1, max_bin=255)
    bst = lgb.train(prm, dtr, int(os.environ.get('ROUNDS', 400)), valid_sets=[dva], callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)])
    bst.save_model(f'{W}/feat/gbdt_{tag}_{tgt}.txt')
    imp = dict(zip(FNAMES, bst.feature_importance('gain'))); tot = sum(imp.values())
    print(tgt, 'best_iter', bst.best_iteration, 'importance(gain %):', {k: round(100 * v / tot, 1) for k, v in sorted(imp.items(), key=lambda x: -x[1])}, flush=True)
