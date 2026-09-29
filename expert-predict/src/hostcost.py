"""Host cost (numpy, 1 thread) of each predictor's per-step update and per-refresh top-51 selection at [75,256]."""
import os; os.environ['OMP_NUM_THREADS'] = '1'
import time, numpy as np, torch
NL, NE = 75, 256; rng = np.random.default_rng(0); fx = np.zeros((NL, NE), bool); fx[:, :26] = True
c = (rng.random((NL, NE)) < 8 / 256).astype(np.float64); sc = np.zeros((NL, NE)); a = 0.5 ** (1 / 512)
def t(f, n=2000):
    f(); t0 = time.perf_counter()
    for _ in range(n): f()
    return (time.perf_counter() - t0) / n * 1e6
def step_base():
    global sc; sc = sc * a + c
E2 = [np.zeros((NL, NE)) for _ in range(3)]
def step_state():   # global + think-clock + answer-clock (only the current state's array decays/accumulates)
    E2[0] = E2[0] * a + c; E2[1] = E2[1] * a + c
def sel(s):
    x = np.where(fx, -np.inf, s); top = np.argsort(-x, 1, kind='stable')[:, :51]
    w = np.zeros((NL, NE), bool); np.put_along_axis(w, top, True, 1); return w
def refresh_base(): sel(sc)
pri = rng.random((NL, NE)); wa = 1.0
def refresh_state(): sel(0.5 * E2[0] * (1 - a) + 0.5 * E2[2] / wa + 0.0 * pri)
REAPw = rng.random((NL, NE))
def refresh_reap(): sel(sc * REAPw)
W = rng.standard_normal((NL * NE, 12)).astype(np.float32); X = rng.random((NL, NE, 12)).astype(np.float32)
Wl = rng.standard_normal((NL, 12)).astype(np.float32)
def refresh_linear(): sel(np.einsum('lef,lf->le', X, Wl))
torch.set_num_threads(1); m1 = torch.nn.Linear(NE * 5, 256); m2 = torch.nn.Linear(256, NE); xt = torch.rand(NL, NE * 5)
def refresh_nn():
    with torch.no_grad(): y = m2(torch.relu(m1(xt))).numpy()
    sel(y)
print('per-step EMA update (us): base %.1f  state-aware %.1f' % (t(step_base), t(step_state)))
for n, f in [('ema512', refresh_base), ('answer-state', refresh_state), ('reap-weighted', refresh_reap), ('linear-12feat', refresh_linear), ('nn-lowrank', refresh_nn)]:
    print('refresh %-14s %.1f us' % (n, t(f, 500)))
