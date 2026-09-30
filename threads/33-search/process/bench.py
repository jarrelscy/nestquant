"""serve-cost microbench: one 16-token refresh step, all 75 layers batched, numpy (1 thread) and torch CPU 8 threads."""
import os, time
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from scipy.special import gammaln
Lr, NE, R, K = 75, 256, 128, 4
rng = np.random.default_rng(0)
cnts = rng.poisson(0.5, (Lr, R, NE)).astype(np.float64); scs = cnts * 1.1
logp = np.log(np.full((Lr, R), 1.0 / R)); a0 = np.full(NE, 0.5)
x = np.zeros((Lr, NE)); 
for l in range(Lr): x[l, rng.choice(NE, 60, replace=False)] = rng.integers(1, 5, 60)
sal = x * 1.2
post = np.full((Lr, NE, K), 0.25); A = np.eye(K) * 0.9 + 0.025; lam = np.array([.03, .3, 1.4, 5.])
E = np.zeros((4, Lr, NE)); kf = np.zeros((2, Lr, NE))
def step():
    # BOCPD (uniform prior), counts + salience run stats
    al = a0 + np.concatenate([np.zeros((Lr, 1, NE)), cnts[:, :-1]], 1)
    As = al.sum(2); n = x.sum(1)[:, None]
    m = x > 0
    lp = gammaln(As) - gammaln(As + n) + ((gammaln(al + x[:, None]) - gammaln(al)) * m[:, None]).sum(2)
    lg = np.concatenate([np.logaddexp.reduce(logp, 1)[:, None] + np.log(1 / 32) + lp[:, :1], logp[:, :-1] + np.log(31 / 32) + lp[:, 1:]], 1)
    lg -= np.logaddexp.reduce(lg, 1)[:, None]
    c2 = np.concatenate([np.zeros((Lr, 1, NE)), cnts[:, :-1]], 1) + x[:, None]
    s2 = np.concatenate([np.zeros((Lr, 1, NE)), scs[:, :-1]], 1) + sal[:, None]
    p = np.exp(lg); rl = np.arange(1, R + 1)
    alp = a0 + c2
    rate = np.einsum("lr,lre->le", p, alp / alp.sum(2, keepdims=True)); srate = np.einsum("lr,lre->le", p / rl, s2)
    # HMM forward
    B = np.exp(x[..., None] * np.log(lam) - lam - gammaln(x[..., None] + 1))
    q = (post @ A) * B; q /= q.sum(2, keepdims=True)
    # Hawkes EMAs + Kalman level/slope
    for j, d in enumerate((0.5, 0.84, 0.957, 0.989)):
        E[j] = d * E[j] + x
    return rate, srate, q
step()
for f, nm in ((step, "numpy 1-thread"),):
    t = time.time(); [f() for _ in range(20)]; print(nm, f"{(time.time() - t) / 20 * 1e3:.1f} ms/refresh (75 layers, BOCPD R={R})")
