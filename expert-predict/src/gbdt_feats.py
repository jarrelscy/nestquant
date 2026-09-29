"""GBDT features for (refresh point, layer, expert) candidates at ranks [RLO, RHI) by EMA256 among non-fixed experts.
Refresh every 16 tokens (one G-block). Refresh r = block b: history = blocks < b (tokens < 16b). All features causal."""
import sys, json; sys.path.insert(0, '/data/Jarrel/expert-predict/src'); from feats import *
RLO, RHI, KP = 20, 121, 8
HLS = (32, 128, 256, 512, 2048)
FNAMES = ['ema32', 'ema128', 'ema256', 'ema512', 'ema2048', 'hits16', 'hits64', 'hits256', 'tok_since_hit',
          'mem_cur_state', 'mem_other_state', 'state_answer', 'tok_since_state_change', 'req_pos',
          'usage_prior', 'reap_mean', 'gain_rel', 'layer', 'coact_same16', 'coact_prev16', 'coact_next16', 'coact_same_ema128', 'rank_ema256']
NR = np.array([np.array(fixed_set_json['n_routed'][str(l + L0)], float) for l in range(NL)]) if False else None
_fj = json.load(open('/data/Jarrel/nestquant/threads/22-boundary-experts/fixed_set.json'))
USAGE = np.array([_fj['n_routed'][str(l + L0)] for l in range(NL)], np.float64); USAGE = (USAGE / USAGE.sum(1, keepdims=True) * K).astype(np.float32)
FXB = FIXED.astype(bool)

def coact_tables(tasks, ntok=400000, seed=0):
    """partners[kind][l,e,:KP] (kind 0 same layer, 1 prev layer l-1, 2 next layer l+1), cosine co-occurrence on train decode tokens."""
    rng = np.random.default_rng(seed); exs = []
    for t in tasks:
        ex = load(t)['ex']; exs.append(ex[np.sort(rng.choice(len(ex), min(len(ex), ntok // len(tasks)), replace=False))])
    ex = np.concatenate(exs).astype(np.int64); T = len(ex)
    n = np.zeros((NL, NE)); [np.add.at(n[l], ex[:, l].ravel(), 1) for l in range(NL)]
    Ps = np.zeros((3, NL, NE, KP), np.int64)
    for l in range(NL):
        a = ex[:, l]
        co = np.bincount((a[:, :, None] * NE + a[:, None, :]).ravel(), minlength=NE * NE).reshape(NE, NE).astype(float)
        np.fill_diagonal(co, 0); cs = co / np.sqrt(np.outer(n[l], n[l]) + 1e-9); Ps[0, l] = np.argsort(-cs, 1)[:, :KP]
        if l > 0:
            b = ex[:, l - 1]; co = np.bincount((a[:, :, None] * NE + b[:, None, :]).ravel(), minlength=NE * NE).reshape(NE, NE).astype(float)
            Ps[1, l] = np.argsort(-(co / np.sqrt(np.outer(n[l], n[l - 1]) + 1e-9)), 1)[:, :KP]
        if l < NL - 1:
            b = ex[:, l + 1]; co = np.bincount((a[:, :, None] * NE + b[:, None, :]).ravel(), minlength=NE * NE).reshape(NE, NE).astype(float)
            Ps[2, l] = np.argsort(-(co / np.sqrt(np.outer(n[l], n[l + 1]) + 1e-9)), 1)[:, :KP]
    return Ps

def gen(d, Ps, every=1, want_y=True):
    """yields (b, cand[NL,nc], X[NL*nc, F] f32, y64[NL*nc], y256[NL*nc], E256[NL,NE]) for refresh blocks b % every == 0 (b>=1)."""
    P = prep(d); C, Cth, Can, bst, nans, bsince = P['C'], P['Cth'], P['Can'], P['bst'], P['nans'], P['bsince']
    nb = C.shape[0]; st = P['st']; N = P['N']
    # tokens since last state change (request start or </think>), at block end
    chg = np.r_[True, st[1:] != st[:-1]] | np.r_[True, d['req'][1:] != d['req'][:-1]]
    ci = np.nonzero(chg)[0]; since_chg = np.arange(N) - ci[np.searchsorted(ci, np.arange(N), 'right') - 1]
    bchg = since_chg[np.minimum(np.arange(nb) * G + G - 1, N - 1)]
    ag = [np.float32(0.5 ** (G / h)) for h in HLS]; E = [np.zeros((NL, NE), np.float32) for _ in HLS]
    sa = 0.5 ** (1 / 2048); Et = np.zeros((NL, NE), np.float32); Ea = np.zeros_like(Et); wt = wa = 0.0
    last = np.full((NL, NE), -10 ** 6, np.int64); s64 = np.zeros((NL, NE), np.int32); s256 = np.zeros((NL, NE), np.int32)
    Cf = C.astype(np.int32); li = np.arange(NL)[:, None]
    stat = np.stack([USAGE, REAP.astype(np.float32), GAINREL.astype(np.float32), np.broadcast_to(np.arange(NL, dtype=np.float32)[:, None], (NL, NE))], -1)
    for b in range(1, nb + 1):
        j = b - 1; c = C[j]
        for k in range(len(HLS)): E[k] = E[k] * ag[k] + c
        na = nans[j]; nt = G - na; dt = np.float32(sa ** nt); da = np.float32(sa ** na)
        Et = Et * dt + Cth[j]; wt = wt * dt + nt; Ea = Ea * da + Can[j]; wa = wa * da + na
        last = np.where(c > 0, j, last); s64 += Cf[j]; s256 += Cf[j]
        if j >= 4: s64 -= Cf[j - 4]
        if j >= 16: s256 -= Cf[j - 16]
        if b % every or b >= nb: continue
        e256 = E[2] * ((1 - ag[2]) / G)
        sc = np.where(FXB, -np.inf, e256); order = np.argsort(-sc, 1, kind='stable'); cand = order[:, RLO:RHI]; nc = cand.shape[1]
        s = int(bst[j]); cur, oth = (Et / max(wt, 1e-6), Ea / max(wa, 1e-6)) if s == 0 else (Ea / max(wa, 1e-6), Et / max(wt, 1e-6))
        g = lambda A: np.take_along_axis(A, cand, 1)
        X = np.empty((NL, nc, len(FNAMES)), np.float32)
        for k in range(len(HLS)): X[..., k] = g(E[k]) * ((1 - ag[k]) / G)
        X[..., 5] = g(c); X[..., 6] = g(s64); X[..., 7] = g(s256); X[..., 8] = np.minimum(G * (b - g(last)), 1e5)
        X[..., 9] = g(cur); X[..., 10] = g(oth); X[..., 11] = s; X[..., 12] = bchg[j]; X[..., 13] = bsince[j]
        X[..., 14:18] = np.take_along_axis(stat, cand[..., None], 1)
        pc = Ps[:, li, cand]                                    # [3, NL, nc, KP]
        X[..., 18] = c[li[..., None], pc[0]].sum(-1)
        cp = np.vstack([np.zeros((1, NE), c.dtype), c[:-1]]); cn = np.vstack([c[1:], np.zeros((1, NE), c.dtype)])
        X[..., 19] = cp[li[..., None], pc[1]].sum(-1); X[..., 20] = cn[li[..., None], pc[2]].sum(-1)
        X[..., 21] = (E[1] * ((1 - ag[1]) / G))[li[..., None], pc[0]].sum(-1); X[..., 22] = np.arange(RLO, RHI)[None, :]
        if want_y:
            y64 = C[b:b + 4].sum(0, dtype=np.int32); y256 = C[b:b + 16].sum(0, dtype=np.int32)
            yield b, cand, X.reshape(-1, len(FNAMES)), g(y64).ravel(), g(y256).ravel(), e256, order[:, :RLO], y64
        else:
            yield b, cand, X.reshape(-1, len(FNAMES)), None, None, e256, order[:, :RLO], None
