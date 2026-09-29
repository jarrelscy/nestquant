"""Compact GBDT variants from saved rows (for the <5 ms/refresh host budget)."""
import sys, time, os; sys.path.insert(0, '/data/Jarrel/expert-predict/src'); from gbdt_feats import *
import lightgbm as lgb
src, tag = sys.argv[1], sys.argv[2]; leaves, rounds, lr = int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])
z = np.load(f'{W}/feat/gbdt_rows_{src}.npz'); FE = os.environ.get('FEATS', ','.join(FNAMES)).split(','); cols = [FNAMES.index(f) for f in FE]
Xtr, Xva = z['Xtr'][:, cols], z['Xva'][:, cols]
for tgt in sys.argv[6].split(','):
    yt, yv = z['ytr' + tgt[1:]], z['yva' + tgt[1:]]
    dtr = lgb.Dataset(Xtr, yt.astype(np.float32), feature_name=FE, free_raw_data=False); dva = lgb.Dataset(Xva, yv.astype(np.float32), reference=dtr)
    prm = dict(objective='poisson', learning_rate=lr, num_leaves=leaves, min_data_in_leaf=500, feature_fraction=0.9, bagging_fraction=0.5,
               bagging_freq=1, num_threads=16, verbose=-1)
    bst = lgb.train(prm, dtr, rounds, valid_sets=[dva], callbacks=[lgb.early_stopping(20), lgb.log_evaluation(25)])
    bst.save_model(f'{W}/feat/gbdt_{tag}_{tgt}.txt')
    # val poisson deviance vs EMA-only predictors on the same rows (mean over rows, lower is better)
    def dev(mu, y): mu = np.maximum(mu, 1e-6); return float(np.mean(mu - y * np.log(mu)))
    Fh = int(tgt[1:]); pv = bst.predict(Xva, num_threads=16)
    print(tgt, 'val nll gbdt %.5f ema128 %.5f ema256 %.5f ema512 %.5f' % (dev(pv, yv), dev(Fh * z['Xva'][:, 1], yv), dev(Fh * z['Xva'][:, 2], yv), dev(Fh * z['Xva'][:, 3], yv)), flush=True)
    Xr = Xva[:7575].copy(); t0 = time.time(); [bst.predict(Xr, num_threads=1) for _ in range(20)]; t1 = (time.time() - t0) / 20
    t0 = time.time(); [bst.predict(Xr, num_threads=4) for _ in range(20)]; t4 = (time.time() - t0) / 20
    imp = dict(zip(FE, bst.feature_importance('gain'))); tot = sum(imp.values())
    print(tgt, 'trees', bst.num_trees(), 'predict 7575 rows: 1thr %.2f ms 4thr %.2f ms' % (t1 * 1e3, t4 * 1e3),
          'importance:', {k: round(100 * float(v) / tot, 1) for k, v in sorted(imp.items(), key=lambda x: -x[1])}, flush=True)
