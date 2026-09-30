"""T33a seq arm B: GBDT on v2's 9 features + raw per-block history in multi-resolution bins (learned temporal filter
without hand-picked decays).  Bins (blocks back from the current block, 0 = the block just closed):
  [0] [1] [2] [3] [4,6) [6,8) [8,12) [12,16) [16,24) [24,32) [32,48) [48,64) [64,128)   for log1p(cnt), log1p(saln)
  (per-block means inside a bin; zero before chain start) + chain position.  Causal.
  binhist.py rows CORPUS            -> $WD/bh/CORPUS/L.npy  [nb, 256, 27] f16
  binhist.py train NAME [sub]       (calib-fit minus val chains; tweedie 1.5 on next-64 saln)
  binhist.py eval NAME CORPUS"""
import os, sys, json, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
from multiprocessing import Pool
import numpy as np
import common as C
import t32lib as T

BINS = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 6), (6, 8), (8, 12), (12, 16), (16, 24), (24, 32), (32, 48), (48, 64), (64, 128)]
VALCH = [3, 8, 13, 18, 23, 28]
V2F = ["ema32", "ema128", "mem_cur_state", "tok_since_hit", "hits16", "sema32", "sema128", "sal16", "mps128"]
BH = [f"{k}_b{i}" for k in ("c", "s") for i in range(len(BINS))] + ["pos"]


def hist(M, sg):
    """M [nb,256] -> [nb,256,13] per-block mean over bins (causal, within chain)."""
    out = np.zeros(M.shape + (len(BINS),), np.float32)
    for s, e in sg:
        cs = np.vstack([np.zeros((1, M.shape[1])), np.cumsum(M[s:e], 0)])
        k = np.arange(e - s)
        for j, (a, b) in enumerate(BINS):
            hi = np.clip(k + 1 - a, 0, None); lo = np.clip(k + 1 - b, 0, None)
            out[s:e, :, j] = (cs[hi] - cs[lo]) / (b - a)
    return out


def rows(args):
    corpus, L = args
    bc, bs = C.load_blocks(corpus, L)
    sg = C.segs_of(corpus, bc.shape[0])
    c = np.log1p(bc); s = np.log1p(bs / C.ML[L])
    pos = np.zeros(bc.shape[0], np.float32)
    for a, b in sg:
        pos[a:b] = np.minimum(np.arange(b - a), 128) / 128
    X = np.concatenate([hist(c, sg), hist(s, sg), np.broadcast_to(pos[:, None, None], bc.shape + (1,))], -1)
    os.makedirs(f"{C.WD}/bh/{corpus}", exist_ok=True)
    np.save(f"{C.WD}/bh/{corpus}/L{L}.npy", X.astype(np.float16))
    return L


def v2feats(corpus, L):
    d = np.load(f"{T.OUT}/rows_bandall/{corpus}/L{L}.npz")
    return T.feature_matrix(V2F, corpus, L, band="all", d=d).reshape(d["cand"].shape + (len(V2F),)), d


def target(corpus, L, nb):
    _, bs = C.load_blocks(corpus, L)
    sg = C.segs_of(corpus, nb)
    y = np.full(bs.shape, np.nan, np.float32)
    s = bs / C.ML[L]
    for a, b in sg:
        cs = np.vstack([np.zeros((1, 256)), np.cumsum(s[a:b], 0)])
        k = np.arange(b - a - 4)
        y[a:a + len(k)] = cs[k + 5] - cs[k + 1]
    return y


def trainrows(args):
    L, sub, seed = args
    rng = np.random.default_rng(seed + L)
    X2, d = v2feats("calib-fit", L)
    assert (d["cand"] == np.arange(256)[None]).all() or True
    Xb = np.load(f"{C.WD}/bh/calib-fit/L{L}.npy")
    ci = d["cand"].astype(np.int64)
    Xb = np.take_along_axis(Xb, ci[..., None], 1)
    y = np.take_along_axis(target("calib-fit", L, ci.shape[0]), ci, 1)
    fx = C.FIXED[L]; nf = ~np.isin(ci, fx)
    blk = np.arange(ci.shape[0]) // 512
    isval = np.isin(blk, VALCH)[:, None]
    ok = nf & np.isfinite(y)
    X = np.concatenate([X2, Xb.astype(np.float32)], -1)
    tr = ok & ~isval & (rng.random(ok.shape) < sub)
    va = ok & isval & (rng.random(ok.shape) < sub)
    return X[tr], y[tr], X[va], y[va]


def train(name, sub=0.15):
    import lightgbm as lgb
    t0 = time.time()
    with Pool(10) as p:
        r = p.map(trainrows, [(L, sub, 0) for L in C.LAYERS])
    Xt = np.concatenate([x[0] for x in r]); yt = np.concatenate([x[1] for x in r])
    Xv = np.concatenate([x[2] for x in r]); yv = np.concatenate([x[3] for x in r])
    del r
    print("rows", Xt.shape, Xv.shape, f"{time.time() - t0:.0f}s", flush=True)
    feats = V2F + BH
    sel = [i for i, f in enumerate(feats) if name.startswith("v2only") is False or f in V2F]
    P = dict(objective="tweedie", tweedie_variance_power=1.5, learning_rate=0.1, num_leaves=63, min_data_in_leaf=2000,
             bagging_fraction=0.5, bagging_freq=1, feature_fraction=0.8, seed=0, num_threads=20, verbosity=-1,
             max_bin=255)
    dt = lgb.Dataset(Xt[:, sel], yt, feature_name=[feats[i] for i in sel], free_raw_data=True)
    dv = lgb.Dataset(Xv[:, sel], yv, reference=dt)
    b = lgb.train(P, dt, 600, valid_sets=[dv], callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)])
    os.makedirs(f"{C.WD}/models", exist_ok=True)
    b.save_model(f"{C.WD}/models/{name}.txt")
    print("saved", name, b.best_iteration, f"{time.time() - t0:.0f}s", flush=True)


def scorelayer(args):
    name, corpus, L = args
    import lightgbm as lgb
    b = lgb.Booster(model_file=f"{C.WD}/models/{name}.txt")
    X2, d = v2feats(corpus, L)
    Xb = np.load(f"{C.WD}/bh/{corpus}/L{L}.npy")
    ci = d["cand"].astype(np.int64)
    Xb = np.take_along_axis(Xb, ci[..., None], 1).astype(np.float32)
    X = np.concatenate([X2, Xb], -1)
    fn = b.feature_name(); allf = V2F + BH
    X = X[..., [allf.index(f) for f in fn]]
    S = np.zeros(ci.shape, np.float32)
    np.put_along_axis(S, ci, b.predict(X.reshape(-1, len(fn)), num_threads=2).reshape(ci.shape).astype(np.float32), 1)
    return S


def evaluate(name, corpus):
    with Pool(12) as p:
        Ss = p.map(scorelayer, [(name, corpus, L) for L in C.LAYERS])
    od = f"{C.WD}/scores/{name}_{corpus}"; os.makedirs(od, exist_ok=True)
    for L, S in zip(C.LAYERS, Ss):
        np.save(f"{od}/L{L}.npy", S.astype(np.float16))
    sweep(od, corpus)


def _m(args):
    od, corpus, L, hm, sel = args
    S = np.load(f"{od}/L{L}.npy").astype(np.float32)
    bc, bs = C.load_blocks(corpus, L)
    if sel is not None:
        idx = np.concatenate([np.arange(c * 512, c * 512 + 512) for c in sel])
        S, bc, bs = S[idx], bc[idx], bs[idx]
    return C.metrics(S, L, bc, bs, C.segs_of(corpus if sel is None else "x", S.shape[0]), hm)


def sweep(od, corpus, hms=(0.3, 0.5, 0.7, 1.0), sel=None):
    with Pool(16) as p:
        r = p.map(_m, [(od, corpus, L, hm, sel) for hm in hms for L in C.LAYERS])
    n = len(C.LAYERS)
    out = {}
    for i, hm in enumerate(hms):
        rr = r[i * n:(i + 1) * n]
        out[hm] = (100 * np.mean([x["sal"] for x in rr]), np.mean([x["churn"] for x in rr]))
    print(od.split("/")[-1], "val" if sel else "", "  ".join(f"hm{h}: {s:.2f}/{c:.2f}" for h, (s, c) in out.items()), flush=True)
    return out


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "rows":
        with Pool(10) as p:
            p.map(rows, [(sys.argv[2], L) for L in C.LAYERS])
    elif cmd == "train":
        train(sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 0.15)
    elif cmd == "eval":
        evaluate(sys.argv[2], sys.argv[3])
        if sys.argv[3] == "calib-fit":
            sweep(f"{C.WD}/scores/{sys.argv[2]}_calib-fit", "calib-fit", sel=VALCH)
