"""Uncapped ranking metric: share of future-64 slots [t+16,t+80) served by fixed + top-51 non-fixed of a score."""
import sys; sys.path.insert(0, '/data/Jarrel/expert-predict/src'); from common import *
FX = FIXED.ravel().astype(bool)
def recall(S, Y, st=None):
    S = S.reshape(len(S), NL, NE).astype(np.float32).copy(); Y = Y.reshape(len(Y), NL, NE).astype(np.float32)
    S[:, FIXED.astype(bool)] = -np.inf
    top = np.argpartition(-S, NF, axis=2)[:, :, :NF]
    hit = np.take_along_axis(Y, top, 2).sum((1, 2)) + (Y * FIXED).sum((1, 2)); tot = Y.sum((1, 2))
    return hit, tot
if __name__ == '__main__':
    tags = sys.argv[2].split(',') if len(sys.argv) > 2 else []
    for t in sys.argv[1].split(','):
        z = np.load(f'{W}/feat/{t}_R64.npz'); m = z['n64'] == 4; X = z['X'][m]; Y = z['Y64'][m]; st = z['st'][m]
        res = {}
        for j, h in enumerate((32, 128, 512, 2048)):
            hh, tt = recall(X[:, j], Y); res[f'ema{h}'] = (hh.sum() / tt.sum(), hh[st == 1].sum() / tt[st == 1].sum())
        hh, tt = recall(Y, Y); res['oracle'] = (hh.sum() / tt.sum(), hh[st == 1].sum() / tt[st == 1].sum())
        for tg in tags:
            S = np.load(f'{W}/feat/S_{tg}_{t}.npy')[m]; hh, tt = recall(S, Y); res[tg] = (hh.sum() / tt.sum(), hh[st == 1].sum() / tt[st == 1].sum())
        print(t, {k: (round(float(a), 4), round(float(b), 4)) for k, (a, b) in res.items()}, flush=True)
