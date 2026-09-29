"""Dump refresh-point features for the neural model: X [nref, 4, NL*NE] f16 EMA rates (hl 32,128,512,2048),
state [nref] i8, Y64/Y256 [nref, NL*NE] f16 future rates over [t+16, t+16+F). usage: mkfeat.py TASK R"""
import sys; sys.path.insert(0, '/data/Jarrel/expert-predict/src'); from feats import *
task = sys.argv[1]; R = int(sys.argv[2])
P = prep(load(task)); nb = P['C'].shape[0]; nref = -(-nb // (R // G)) + 1
pr = np.zeros((2, NL, NE), np.float32)
X = np.zeros((nref, 4, NL * NE), np.float16); st = np.zeros(nref, np.int8)
for r, s, x in iter_feats(P, R, pr, hl=(32, 128, 512, 2048), shl=(256, 2048)):
    X[r] = x[..., :4].reshape(-1, 4).T; st[r] = s
out = dict(X=X, st=st)
for F in (64, 256):
    cs, i0, i1 = fut_target(P['C'], R, F, nref, 1)
    y = ((cs[i1] - cs[i0]) / np.maximum(i1 - i0, 1)[:, None, None] / G).reshape(nref, -1)
    out[f'Y{F}'] = y.astype(np.float16); out[f'n{F}'] = (i1 - i0).astype(np.int16)
os.makedirs(f'{W}/feat', exist_ok=True); np.savez(f'{W}/feat/{task}_R{R}.npz', **out); print(task, nref)
