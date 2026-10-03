"""NQ_S3_EXT hook: p-phase GBDT count head (LightGBM, live-computable features), = /data/Jarrel/nq-step3/p-phase/miss/exp_gbdt.py
feature for feature. Use with the s3 arm as the p-online stack 'pick+gbdt<beta>':
    NQ_S3=1 NQ_S3_X=1 NQ_S3_EXT=<repo>/streaming/s3x_gbdt.py:cnt NQ_S3_XPOS=1 NQ_S3_XW=<beta/(1+beta)> NQ_S3_XRESET=0
(beta .35 -> xw 0.259259; xpos=1 = v <- (nr(v) + beta nr(max(cnt, 0))) * rowsum(v) up to a common factor; xreset=0 because
the rolling counts run across request boundaries.)
Features per (l, e) at a refresh at global decode row t (rows < t), all counted over decode rows since this hook was
loaded (boot / ctl s3xf change), across requests:
    layer, jF S (raw), -min(t - lastuse, 8192), c16 / c64 / c256 / c1024 (activations in the last 16/64/256/1024 rows),
    haz = -(rows since last use / mean inter-arrival) if > 1 use else -1e3, prior (static 75x256 table)
Returns None (= base policy) for t < NQ_S3X_GB_TMIN (256; the heads were trained on t >= 256).
env: NQ_S3X_GB_DIR (dir with gb_<head>.txt + gb_prior.npy; default <this dir>/s3x), NQ_S3X_GB_HEAD (pcnt_x6 | pcnt_s_x6),
     NQ_S3X_GB_THREADS (2; LightGBM threads, the serve CPU is shared with the I/O engine), NQ_S3X_GB_F16 (1 = round the output to fp16 like the stack exports; tests only),
     NQ_S3X_GB_ASYNC (1 = LightGBM runs in a worker thread (ctypes drops the GIL) and a refresh gets the prediction made
     from the PREVIOUS refresh's features (one refresh = 16 rows late, deterministic: waits if not done); the host loop
     then only pays the features, ~2-4 ms, instead of features + predict; 0 = predict inline, = the p-online sim exactly).
     Host-loop cost (CPU replay paced at 80 tok/s, tb-cad-model, loaded box): tap step at a refresh p50 1.3 ms (arm alone) ->
     3.7 ms async 2 threads (predict ~70 ms in the worker, never waited on); sync 4 threads 59 ms, pcnt_s sync 2 threads 26 ms.
     Sim cost of the 1-refresh lag vs exact (pick+gbdt.35): kldD-w0 -0.04/-0.06, cad -0.01/-0.10, chinese -0.12/-0.17.
cost: state['us'] = [calls, seconds total, seconds features, seconds waiting on the async predict] (sync: total includes the predict).
Needs a history ring >= 1024 rows (NQ_S3_XRING, default 1024) and lightgbm (nq_vllm puts NQ_LGB_PATH on sys.path)."""
import os, time
import numpy as np

NL, NE = 75, 256
WS = (16, 64, 256, 1024)
_M = {}


def _model(st):
    d = os.environ.get('NQ_S3X_GB_DIR') or os.path.join(os.path.dirname(os.path.abspath(__file__)), 's3x')
    h = os.environ.get('NQ_S3X_GB_HEAD', 'pcnt_x6'); k = (d, h)
    if k not in _M:
        import lightgbm as lgb
        _M[k] = (lgb.Booster(model_file=f'{d}/gb_{h}.txt'), np.load(f'{d}/gb_prior.npy').astype(np.float32).ravel())
    return _M[k]


def _init(st):
    st.update(t=0, last=np.full((NL, NE), -10 ** 6, np.int64), first=np.full((NL, NE), -1, np.int64), tot=np.zeros((NL, NE), np.float32),
              lay=np.repeat(np.arange(NL), NE).astype(np.float32), ar=np.arange(NL)[:, None], thr=int(os.environ.get('NQ_S3X_GB_THREADS', '2')),
              tmin=int(os.environ.get('NQ_S3X_GB_TMIN', '256')), f16=os.environ.get('NQ_S3X_GB_F16', '0') == '1', us=[0, 0.0, 0.0, 0.0],
              asy=os.environ.get('NQ_S3X_GB_ASYNC', '1') == '1', fut=None)


def feats(cx):
    """advance the all-time state by the new rows, return X [NL*NE, 9] float32 for the refresh at row t (= rows seen)"""
    st = cx.state
    if 't' not in st: _init(st)
    ids = cx.ids; n = len(ids); k = min(cx.nnew, n); ar = st['ar']; last = st['last']; first = st['first']; tot = st['tot']
    for i in range(n - k, n):
        r = ids[i].astype(np.int64); t = st['t']; last[ar, r] = t; f = first[ar, r]; first[ar, r] = np.where(f < 0, t, f); tot[ar, r] += 1; st['t'] = t + 1
    t = st['t']; flat = (ids.astype(np.int64) + (np.arange(NL) * NE)[None, :, None]).reshape(n, -1)
    C = []; acc = np.zeros(NL * NE, np.float32); lo = n
    for w in WS:                                            # nested windows: c_w = c_prev + counts of rows [n-w, n-w_prev)
        a = max(n - w, 0)
        if a < lo: acc = acc + np.bincount(flat[a:lo].ravel(), minlength=NL * NE).astype(np.float32); lo = a
        C.append(acc)
    j = np.asarray(cx.S, np.float32).ravel(); recn = np.minimum(t - last, 8192).astype(np.float32)
    ia = np.where(first >= 0, t - first, 0) / np.maximum(tot, 1); haz = -np.where(tot > 1, recn / np.maximum(ia, 1), 1e3)
    return t, np.column_stack([st['lay'], j, -recn.ravel()] + C + [haz.ravel(), st['pri']]).astype(np.float32)


def cnt(cx):
    st = cx.state
    if 't' not in st: _init(st)
    a = time.perf_counter(); B, st['pri'] = _model(st)
    t, X = feats(cx); st['us'][2] += time.perf_counter() - a
    if t < st['tmin'] or len(cx.ids) < min(t, 1024): st['fut'] = None; return None   # ring shorter than the 1024 window (reset / too small)
    if st['asy']:
        if 'ex' not in st:
            import concurrent.futures as cf; st['ex'] = cf.ThreadPoolExecutor(1, thread_name_prefix='nq-s3x-gb')
        prev = st['fut']; st['fut'] = st['ex'].submit(B.predict, X, num_threads=st['thr'])
        if prev is None: return None
        w = time.perf_counter(); y = prev.result().reshape(NL, NE); st['us'][3] += time.perf_counter() - w
    else:
        y = B.predict(X, num_threads=st['thr']).reshape(NL, NE)
    if st['f16']: y = y.astype(np.float16)
    st['us'][0] += 1; st['us'][1] += time.perf_counter() - a
    return y.astype(np.float32)
