"""oracle decision diagnostics on heldout: future-sum oracle at horizons n blocks, with mult hysteresis / additive / cap."""
import sys, numpy as np
from multiprocessing import Pool
import dlib as D
corpus = sys.argv[1] if len(sys.argv) > 1 else "glm52-heldout"

def fut(M, n):
    out = np.zeros(M.shape)
    for c0 in range(0, M.shape[0], D.NBC):
        s = M[c0:c0 + D.NBC]
        cs = np.vstack([np.zeros((1, D.NE)), np.cumsum(s, 0)])
        k = np.arange(s.shape[0])
        out[c0:c0 + D.NBC] = cs[np.minimum(k + 1 + n, s.shape[0])] - cs[k + 1]
    return out

ARMS = []
for n in (1, 4, 8, 16, 32):
    for hm in (0, 0.25, 0.5, 1, 2, 4):
        ARMS.append((n, "mult", hm))
    for cap in (1, 2, 3, 4):
        ARMS.append((n, "cap", cap))

def job(L):
    bs, bc = D.load_blk(corpus, L); fx = D.fx_mask(L); fd = D.fd_mask(L)
    m = bs.sum() / bc.sum()
    F = {n: fut(bs, n) / m for n in (1, 4, 8, 16, 32)}
    out = []
    for n, kind, p in ARMS:
        S = F[n] + 1e-6 * F[32]
        if kind == "mult":
            sv = D.replay(S, S * (1 + p) + 1e-9, fx, fd, D.NBC, 0)
        else:
            sv = D.replay(S, S, fx, fd, D.NBC, int(p))
        out.append(D.evaluate(sv, bs, fx))
    return out

with Pool(16) as p: R = np.array(p.map(job, D.LAYERS))
M = R.mean(0)
for i, a in enumerate(ARMS):
    print(f"oracle next-{a[0]*16:4d} {a[1]} {a[2]:5}  sal {M[i,0]*100:.2f}  churn {M[i,1]:.2f}")
