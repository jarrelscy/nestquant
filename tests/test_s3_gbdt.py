"""NQ_S3_EXT GBDT hook (streaming/s3x_gbdt.py:cnt) vs the p-phase export + p-online stack 'pick+gbdt<beta>' (p-online
t_ab2.py class X: v = (nr(v_pick) + beta nr(max(cnt[t], 0))) * rowsum(v_pick), t >= 256), CPU only.
The live TapScheduler is driven as in test_s3_adm.py (olib.budget_sim policy, jF replay, state across requests):
  env NQ_S3=1 NQ_S3_RESET=0 NQ_S3_X=1 NQ_S3_XPOS=1 NQ_S3_XW=beta/(1+beta) NQ_S3_XRESET=0 NQ_S3_EXT=s3x_gbdt.py:cnt
  1. the hook's prediction at every refresh row t >= 256 must equal the p-phase export stack/gbdt/<stream>.npz 'cnt'[t]
     (NQ_S3X_GB_F16=1 rounds to the export's fp16; features + LightGBM are recomputed live from the history ring)
  2. live keys = X.score at every refresh (rel 1e-5), 3. budget_sim = res_ab2_<stream>.json 'pick+gbdt<beta>' when present
  ASYNC=1: NQ_S3X_GB_ASYNC=1, the hook's value at t must be the export at t - 16 and X uses cnt[t - 16] (from t >= 272)
  python tests/test_s3_gbdt.py [stream ...]  (kldD-<w>_r0 | tb-<task> | gen-<key>; default kldD-w0_r0; BETA=.35, HEAD=pcnt_x6)"""
import os, sys, json, time, zipfile
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import test_s3_adm as A
from test_s3_adm import OL, PL, PO
SD = '/data/Jarrel/nq-step3/stack/gbdt'
PICK = dict(b=.25, a=.9, p=.25, mass='512', wL=.5, kL=.5, nb=22)
A.S3ENV = A.S3ENV + ('NQ_S3_X', 'NQ_S3_XW', 'NQ_S3_EXT', 'NQ_S3_XRING', 'NQ_S3_XPOS', 'NQ_S3_XRESET')


def npz_mm(path, key):
    """memmap of an uncompressed (ZIP_STORED) npz member"""
    with zipfile.ZipFile(path) as zf:
        info = zf.getinfo(key + '.npy')
        with zf.open(info) as fh:
            ver = np.lib.format.read_magic(fh); shape, fo, dt = (np.lib.format.read_array_header_1_0 if ver == (1, 0) else np.lib.format.read_array_header_2_0)(fh); hl = fh.tell()
    with open(path, 'rb') as f:
        f.seek(info.header_offset); h = f.read(30); n, m = int.from_bytes(h[26:28], 'little'), int.from_bytes(h[28:30], 'little')
    return np.memmap(path, dt, 'r', offset=info.header_offset + 30 + n + m + hl, shape=shape)


def nr(x): m = x.sum(-1, keepdims=True); return x / np.where(m > 0, m, 1)


class X(PL.Combo):
    """p-online t_ab2.X with mix=[(cnt, beta)], tmin=256"""
    def __init__(s, st, cnt, beta, lag=0, **kw): super().__init__(st, **kw); s.cnt = cnt; s.beta = beta; s.lag = lag
    def score(s, t):
        v, _ = super().score(t)
        if t < 256 + 16 * s.lag: return v, v
        m = v.sum(1, keepdims=True); v = (nr(v) + s.beta * nr(np.maximum(np.asarray(s.cnt[t - 16 * s.lag], np.float32), 0))) * m
        return v, v


def main():
    names = sys.argv[1:] or ['kldD-w0_r0']; beta = float(os.environ.get('BETA', '.35')); ok = True
    lag = int(os.environ.get('ASYNC', '0') == '1'); os.environ['NQ_S3X_GB_ASYNC'] = str(lag); os.environ['NQ_S3X_GB_F16'] = '1'; os.environ['NQ_S3X_GB_HEAD'] = os.environ.get('HEAD', 'pcnt_x6')
    for nm in names:
        st = OL.load_kldD(nm[5:], hp=False) if nm.startswith('kldD-') else A.load(nm)
        cnt = npz_mm(f'{SD}/{st["name"]}.npz', 'cnt_s' if 'pcnt_s' in os.environ['NQ_S3X_GB_HEAD'] else 'cnt'); assert cnt.shape[0] == len(st['ids'])
        T = A.boot(st['S'], dict(NQ_S3='1', NQ_S3_RESET='0', NQ_S3_X='1', NQ_S3_XPOS='1', NQ_S3_XW=repr(beta / (1 + beta)), NQ_S3_XRESET='0',
                                 NQ_S3_EXT=HERE + '/../streaming/s3x_gbdt.py:cnt'))
        chk = dict(n=0, bad=0, worst=0.0, none=0); orig = T._s3xcall
        def cap():
            y = orig(); t = T.tok
            if y is None: chk['none'] += 1; return y
            e = np.asarray(cnt[t - 16 * lag], np.float32); d = float(np.abs(y - e).max()); chk['n'] += 1; chk['worst'] = max(chk['worst'], d); chk['bad'] += d > 0
            return y
        T._s3xcall = cap
        r, L, dt = A.run(st, T, X(st, cnt, beta, lag, **PICK), None)
        good = L.bad == 0 and L.n > 0 and chk['bad'] == 0 and chk['n'] + chk['none'] == L.n and T.s3xbad == 0
        fn = f'{PO}/res_ab2_{st["name"]}.json'; k = f'pick+gbdt{beta}' if 'pcnt_s' not in os.environ['NQ_S3X_GB_HEAD'] else f'pick+gbdt_s{beta}'
        line = (f'{st["name"]} {k}: hook = export on {chk["n"]} refreshes ({chk["bad"]} differ, max |d| {chk["worst"]:.3g}; {chk["none"]} t<256 -> base); '
                f'keys vs X {L.n} refreshes max rel {L.worst:.1e}; live {r[0]:.2f}/{r[1]:.2f}')
        if lag: k += ' async'
        if os.path.exists(fn) and k in json.load(open(fn)):
            e = json.load(open(fn))[k]['all']; dd = (r[0] - e[0], r[1] - e[1]); good &= max(abs(dd[0]), abs(dd[1])) <= 0.02
            line += f'; p-online {e[0]:.2f}/{e[1]:.2f} (d {dd[0]:+.3f}/{dd[1]:+.3f})'
        st_ = T.s3xstate; u = st_['us']; line += f'; hook {1e3 * T.s3xt[1] / max(T.s3xt[0], 1):.1f} ms/call (feats {1e3 * u[2] / max(T.s3xt[0], 1):.1f}; {st_["thr"]} thr)  [{dt:.0f}s] ' + ('ok' if good else 'FAIL')
        print(line, flush=True); ok &= good
    return ok


if __name__ == '__main__':
    ok = main(); print('PASS' if ok else 'FAIL'); sys.exit(0 if ok else 1)
