"""per-layer slot allocation from routing concentration (train tasks only: fin-saccr-rwa, pretrain-shard-corruption).
cov_l(k) = mean over 256-token windows of the hit share of the window's k most-used experts in layer l (each window's
own top-k; EMA512-predicted top-k for the 'ema' variant).  Greedy water-filling of the same total (77 x 75 = 5775) on the
marginal coverage gain, k_l in [kmin, kmax].  -> nqalgo/slots_<name>.json {"nf": [75 ints]}"""
import numpy as np, json, sys
sys.path.insert(0, '/data/Jarrel/nq-tfpred/src'); import data as D
W = 16; TOT = 77 * 75
cov = {'orc': np.zeros((75, 257)), 'ema': np.zeros((75, 257))}; n = 0
for t in ['fin-saccr-rwa', 'pretrain-shard-corruption']:
    c = D.ids_blocks(t)['cnt']; nb = len(c) // W * W                  # mmap u8 [nb,75,256]
    E = np.zeros((75, 256), np.float32); a = 0.5 ** (16 / 512); prev = None; nw = nb // W; ne = 0
    for w0 in range(nw):
        blk = np.asarray(c[w0 * W:(w0 + 1) * W], np.float32)
        win = blk.sum(0); tot = win.sum(-1, keepdims=True).clip(1)
        so = -np.sort(-win, -1); cov['orc'][:, 1:] += so.cumsum(-1) / tot
        if prev is not None:                                          # EMA state at previous window end predicts this window
            g = np.take_along_axis(win, np.argsort(-prev, -1), -1); cov['ema'][:, 1:] += g.cumsum(-1) / tot; ne += 1
        for b in range(W): E = E * a + blk[b]
        prev = E.copy()
    cov['ema'] *= 1.0  # accumulated over ne windows; rescale below
    n += nw; ne_tot = globals().get('ne_tot', 0) + ne; globals()['ne_tot'] = ne_tot
cov['ema'] *= n / ne_tot
for k in cov: cov[k] /= n
for name, kmin, kmax in [('orc', 40, 120), ('ema', 40, 120)]:
    cv = cov[name]; nf = np.full(75, kmin)
    while nf.sum() < TOT:
        g = np.array([cv[l, nf[l] + 1] - cv[l, nf[l]] if nf[l] < kmax else -1 for l in range(75)]); nf[np.argmax(g)] += 1
    u = cv[np.arange(75), 77].mean(); v = cv[np.arange(75), nf].mean()
    json.dump(dict(nf=nf.tolist(), cov_uniform77=float(u), cov_alloc=float(v), source=name), open(f'/data/Jarrel/nq-tfpred/nqalgo/slots_{name}.json', 'w'))
    print(name, 'uniform77 cov %.4f -> alloc %.4f' % (u, v), 'nf min/max', nf.min(), nf.max(), nf.tolist())
