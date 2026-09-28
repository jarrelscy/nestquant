"""Step 3/4: LDLQ + Lloyd-Max scalar with per-block (and per block x row-group) rates.
Compares uniform vs allocated at equal total bits on proxy loss and expert-output error."""
import argparse, json, math, sys, time, heapq
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, '/home/coder/git/nestquant/threads/01-rate-allocation')
sys.path.insert(0, '/home/coder/git/orbit-duet')
from common import *
from orbit_duet.source import weights as load_weights

p = argparse.ArgumentParser()
p.add_argument('--model', default='glm'); p.add_argument('--layer', type=int, default=16); p.add_argument('--expert', type=int, default=36)
p.add_argument('--rates', default='2,3,4'); p.add_argument('--sample-rows', type=int, default=8192)
p.add_argument('--schemes', default='uniform,blk_int,blk_half,blk_fine,blk_int_fb,blk_fine_fb,row_int,2d_int,2d_fine,actorder_uniform,actorder_blk_fine')
p.add_argument('--eval', action='store_true'); p.add_argument('--adapt-scale', type=int, default=1); p.add_argument('--damp', type=float, default=0.025); p.add_argument('--tfile', default=None); p.add_argument('--had-down', type=int, default=128); p.add_argument('--out', required=True)
a = p.parse_args()
setup_gpu(); dev = 'cuda'
torch.manual_seed(0)

# ---------------- Lloyd-Max codebooks for unit Gaussian, N = 1..256 levels ----------------
def lloyd_max(N, iters=300):
    if N == 1: return np.array([0.0]), 1.0
    x = np.linspace(-9, 9, 360001); w = np.exp(-x * x / 2); w /= w.sum()
    c = np.quantile(np.random.default_rng(0).standard_normal(400000), (np.arange(N) + 0.5) / N)
    for _ in range(iters):
        b = (c[1:] + c[:-1]) / 2
        idx = np.searchsorted(b, x)
        s = np.bincount(idx, w * x, N); m = np.bincount(idx, w, N)
        c = np.where(m > 0, s / np.maximum(m, 1e-300), c)
    b = (c[1:] + c[:-1]) / 2; idx = np.searchsorted(b, x)
    return c, float((w * (x - c[idx]) ** 2).sum())

NMAX = 256
import os, pickle
_cache = '/tmp/nestquant/01-rate-allocation/lloyd_max.pkl'
if os.path.exists(_cache):
    CB, DN = pickle.load(open(_cache, 'rb'))
else:
    CB, DN = {}, np.zeros(NMAX + 1)
    for N in list(range(1, 65)) + [128, 256]:
        CB[N], DN[N] = lloyd_max(N, 200 if N <= 64 else 100)
    pickle.dump((CB, DN), open(_cache, 'wb'))
LEVELS = sorted(CB)  # allowed level counts for "fine" schemes: 1..64,128,256
CBT = {N: torch.tensor(c, dtype=torch.float32, device=dev) for N, c in CB.items()}
BND = {N: (c[1:] + c[:-1]) / 2 for N, c in CBT.items()}

def qnearest(x, N):
    if N == 1: return torch.zeros_like(x)
    return CBT[N][torch.searchsorted(BND[N], x.contiguous())]

# ---------------- allocation ----------------
def alloc_greedy(a, Rbar, allowed):
    """Greedy by marginal return per bit over an allowed ladder of level counts (prefix-nested by construction).
    a: importance weights (flattened units of equal size). Returns level counts per unit, total bits == Rbar*len exactly-ish."""
    a = np.asarray(a, np.float64).ravel(); m = len(a)
    allowed = sorted(allowed); pos = np.zeros(m, int)
    budget = Rbar * m - m * math.log2(allowed[0])
    def gain(i):
        if pos[i] + 1 >= len(allowed): return None
        n0, n1 = allowed[pos[i]], allowed[pos[i] + 1]
        return a[i] * (DN[n0] - DN[n1]) / math.log2(n1 / n0), math.log2(n1 / n0)
    h = []
    for i in range(m):
        g = gain(i)
        if g: h.append((-g[0], i, g[1]))
    heapq.heapify(h); spent = 0.0
    while h:
        ng, i, c = heapq.heappop(h)
        if spent + c > budget + 1e-9: continue  # skip; smaller steps may still fit
        spent += c; pos[i] += 1
        g = gain(i)
        if g: heapq.heappush(h, (-g[0], i, g[1]))
    return np.array([allowed[k] for k in pos])

INT = [2 ** r for r in range(0, 9)]            # integer bits 0..8
HALF = sorted(set([1, 2, 3, 4, 6, 8, 11, 16, 23, 32, 45, 64, 128, 256]) & set(LEVELS)) # ~half-bit ladder (<=6 then int)

# ---------------- LDLQ ----------------
@torch.no_grad()
def ldlq(W, Lt, Nmap, order_perm=None):
    """W (k,n) normalized rotated weights; Lt unit block-lower (k,k); Nmap: (m, n) level counts per (block, column).
    Quantize blocks from last to first with error feedback (EXL3 order). Returns Q, per-block target var."""
    k, n = W.shape; m = k // 16
    Q = torch.zeros_like(W); E = torch.zeros_like(W); tv = np.zeros(m)
    Nm = torch.as_tensor(Nmap, device=dev)
    for i in range(m - 1, -1, -1):
        s, e = 16 * i, 16 * (i + 1)
        tgt = W[s:e] + (Lt[e:, s:e].T @ E[e:] if e < k else 0)
        tv[i] = float(tgt.square().mean())
        sc = math.sqrt(tv[i]) if a.adapt_scale else 1.0   # per-block fp16 scale (metadata ~0.0005 bpw)
        sc = float(torch.tensor(sc).half())
        tgt = tgt / sc
        q = torch.empty_like(tgt)
        row = Nm[i]
        for N in torch.unique(row).tolist():
            cols = (row == N).nonzero().flatten()
            q[:, cols] = qnearest(tgt[:, cols], N)
        q = q * sc
        Q[s:e] = q; E[s:e] = W[s:e] - q
    return Q, tv

# ---------------- load ----------------
src = GLM_SRC if a.model == 'glm' else MIMO_SRC
g, u, d = load_weights(src, a.layer, a.expert, device=dev)
Hx, Hh, outs, meta = load_grams(a.model, a.layer, a.expert)
Hx = Hx.to(dev, torch.float64); Hh = Hh.to(dev, torch.float64)
dn2 = d.double().square().sum(0)  # ||down[:, r]||^2, r over intermediate
row_imp = {'gate': (outs[0].diagonal().to(dev).double() * dn2).cpu().numpy(),
           'up': (outs[1].diagonal().to(dev).double() * dn2).cpu().numpy(),
           'down': np.ones(6144)}
projs = {'gate': (g, Hx, 11), 'up': (u, Hx, 11), 'down': (d, Hh, 12)}
report = dict(expert=f'{a.model}_l{a.layer}_e{a.expert}', rowimp_amgm_db={}, proj={}, output={})
for k_, v in row_imp.items():
    report['rowimp_amgm_db'][k_] = dict(per_row=amgm_db(v), per16=amgm_db(v.reshape(-1, 16).mean(1)), per128=amgm_db(v.reshape(-1, 128).mean(1)))
print(json.dumps(report['rowimp_amgm_db']), flush=True)

prep = {}
def aomode(s):
    return 'sort' if s.startswith('actorder') else ('shard' if s.startswith('ashard') else '')
for name, (w, H, seed) in projs.items():
    modes = {''} | {aomode(s) for s in a.schemes.split(',')}
    for ao in modes:
        Hp = H; perm = None
        if ao == 'sort' or (ao == 'shard' and name != 'down'):
            perm = torch.argsort(H.diagonal())  # ascending: largest-diag channels at the end = quantized first
        if ao == 'shard' and name == 'down':
            # TP8-compatible: intermediate-dim permutation is free (absorbed into gate/up rows); deal sorted channels
            # round-robin to 8 shards so every shard has the same mix, sort ascending within shard
            srt = torch.argsort(H.diagonal()).cpu()
            perm = torch.cat([srt[j::8] for j in range(8)]).to(dev)
        if ao == 'shard' and name != 'down':
            perm = None
        if perm is not None:
            Hp = H[perm][:, perm]
        HAD = a.had_down if name == 'down' else 128
        Hr, su = rotate_H(Hp, seed=seed, damp=a.damp, had=HAD)
        Lt, D = block_ldl(Hr, 16)
        t = (D.diagonal(dim1=1, dim2=2).sum(-1) / 16).cpu().numpy()
        if a.tfile and ao == '':  # externally supplied (e.g. cross-validated) allocation weights
            t = np.load(a.tfile.format(proj='down' if name == 'down' else 'gate_up'))
        WT = w.T.double()
        if perm is not None: WT = WT[perm]
        Wr = rotate_W_in(WT, su, had=HAD)
        sv = Wr.square().mean(0).sqrt()  # per output column scale (free in format, like EXL3 sv)
        Hr_true, _ = rotate_H(Hp, seed=seed, damp=0.0, had=HAD)
        prep[(name, ao)] = dict(had=HAD, Lt=Lt.float(), t=t, Wn=(Wr / sv).float(), sv=sv, su=su, perm=perm, Htrue=Hr_true.float(),
                                den=float(((Hr_true @ Wr) * Wr).sum()))
        del Hr, D

def colgroups(name, gsize):
    v = row_imp[name]; return v.reshape(-1, gsize).mean(1)

def make_Nmap(scheme, name, Rbar, t_eff):
    m = len(t_eff); n = 2048 if name != 'down' else 6144
    if scheme.endswith('_tp') and name == 'down':  # fixed bytes per TP8 shard (16 input blocks each)
        base = scheme[:-3]
        parts = [make_Nmap(base, name, Rbar, t_eff[j:j + 16]) for j in range(0, m, 16)]
        return np.concatenate(parts, 0)
    if scheme.endswith('_tp'): scheme = scheme[:-3]
    for pre in ('ashard_', 'actorder_'):
        if scheme.startswith(pre) and scheme != 'actorder_uniform': scheme = scheme[len(pre):]
    if scheme == 'ashard_uniform': scheme = 'uniform'
    if scheme in ('uniform', 'actorder_uniform'):
        return np.full((m, n), 2 ** Rbar)
    if scheme in ('blk_int', 'blk_int_fb'):
        N = alloc_greedy(t_eff, Rbar, INT); return np.repeat(N[:, None], n, 1)
    if scheme == 'blk_half':
        N = alloc_greedy(t_eff, Rbar, HALF); return np.repeat(N[:, None], n, 1)
    if scheme in ('blk_fine', 'blk_fine_fb', 'actorder_blk_fine'):
        N = alloc_greedy(t_eff, Rbar, LEVELS); return np.repeat(N[:, None], n, 1)
    gs = 128
    b = colgroups(name, gs) if name != 'down' else np.ones(n // gs)
    b = b / b.mean()
    if scheme == 'row_int':
        N = alloc_greedy(b, Rbar, INT); return np.repeat(np.repeat(N[None, :], m, 0), gs, 1)
    if scheme in ('2d_int', '2d_fine'):
        A = np.outer(t_eff, b)
        N = alloc_greedy(A.ravel(), Rbar, INT if scheme == '2d_int' else LEVELS).reshape(m, -1)
        return np.repeat(N, gs, 1)
    raise ValueError(scheme)

def bits_of(Nmap): return float(np.log2(Nmap).mean())

@torch.no_grad()
def dequant(name, ao, Q):
    P = prep[(name, ao)]
    Wq = unrotate_W_in(Q.double() * P['sv'], P['su'], had=P['had'])
    if P['perm'] is not None:
        inv = torch.empty_like(P['perm']); inv[P['perm']] = torch.arange(len(inv), device=dev)
        Wq = Wq[inv]
    return Wq.T.float().contiguous()

# sample rows
S = torch.load(stats_path(a.model, a.layer, a.expert) + '_training_sample.pt', weights_only=True, mmap=True)
nrow = min(a.sample_rows, len(S['x']))
sel = torch.randperm(len(S['x']), generator=torch.Generator().manual_seed(1))[:nrow]
Xs = S['x'][sel].to(dev); Ps = S['p'][sel].to(dev).double()

def expert_out(x, gw, uw, dw):
    return F.linear(F.silu(F.linear(x.float(), gw)) * F.linear(x.float(), uw), dw)

@torch.no_grad()
def out_err(x, pr, ws, ref_ws):
    num = den = 0.0; pnum = pden = 0.0
    for i in range(0, len(x), 1024):
        xb = x[i:i + 1024]; y = expert_out(xb, *ref_ws).double(); yq = expert_out(xb, *ws).double()
        e = (yq - y).square().sum(-1); en = y.square().sum(-1); p2 = pr[i:i + 1024].square()
        num += float((e * p2).sum()); den += float((en * p2).sum()); pnum += float(e.sum()); pden += float(en.sum())
    return math.sqrt(num / den), math.sqrt(pnum / pden)

ref = (g.float(), u.float(), d.float())
rates = [int(r) for r in a.rates.split(',')]
schemes = a.schemes.split(',')
deq_store = {}
for Rbar in rates:
    for scheme in schemes:
        ao = aomode(scheme)
        res = {}; ws = {}
        for name in ['gate', 'up', 'down']:
            P = prep[(name, ao)]; t_eff = P['t']
            if scheme.endswith('_fb'):
                # feedback-aware: measure per-block target variance under the uniform-rate run, then reallocate
                N0 = make_Nmap('blk_int' if 'int' in scheme else 'blk_fine', name, Rbar, t_eff)
                _, tv = ldlq(P['Wn'], P['Lt'], N0)
                t_eff = t_eff * tv
            Nmap = make_Nmap(scheme, name, Rbar, t_eff)
            t0 = time.time()
            Q, tv = ldlq(P['Wn'], P['Lt'], Nmap)
            Er = (P['Wn'] - Q).double() * P['sv']
            proxy = float(((P['Htrue'].double() @ Er) * Er).sum()) / P['den']
            ws[name] = dequant(name, ao, Q)
            res[name] = dict(bits=bits_of(Nmap), proxy=proxy, proxy_db=10 * math.log10(proxy),
                             tv_first_last=[float(tv[:8].mean()), float(tv[-8:].mean())], secs=time.time() - t0)
            if scheme in ('blk_int', 'blk_fine', 'blk_half', 'uniform'):
                res[name]['block_rates'] = np.log2(Nmap[:, 0]).round(3).tolist()
        r, f = out_err(Xs, Ps, (ws['gate'], ws['up'], ws['down']), ref)
        res['out_routed_train'] = r; res['out_plain_train'] = f
        deq_store[(Rbar, scheme)] = {k: v.cpu() for k, v in ws.items()}
        report['proj'].setdefault(str(Rbar), {})[scheme] = res
        print(f'R={Rbar} {scheme:18s} ' + ' '.join(f"{n}:{res[n]['bits']:.3f}b {res[n]['proxy_db']:.3f}dB" for n in ['gate', 'up', 'down']) +
              f"  out_routed {100*r:.3f}%  plain {100*f:.3f}%", flush=True)
        json.dump(report, open(a.out, 'w'), indent=1)

if a.eval and a.model == 'glm':
    caps = ['native_id_control_v1_capture', 'ood_controlled_v1_capture', 'glm53_matched_context_pilot_v1_capture']
    for c in caps:
        C = torch.load(f'{ROOT}/runs/{c}/layer_{a.layer}.pt', weights_only=True, mmap=True)
        routed, slots = torch.where(C['ids'] == a.expert)
        X_all = C['x'].to(dev)
        for key, ws in deq_store.items():
            wsd = tuple(ws[n].to(dev) for n in ['gate', 'up', 'down'])
            forced = out_err(X_all, torch.ones(len(X_all), device=dev, dtype=torch.float64), wsd, ref)[1]
            rr = out_err(C['x'][routed].to(dev), C['p'][routed, slots].to(dev).double(), wsd, ref)[0] if len(routed) else None
            report['output'].setdefault(c, {}).setdefault(str(key[0]), {})[key[1]] = dict(forced=forced, routed=rr, routed_rows=len(routed))
        print(c, 'routed rows', len(routed), {f'{k[0]}:{k[1]}': round(100 * report['output'][c][str(k[0])][k[1]]['forced'], 3) for k in deq_store}, flush=True)
        del X_all
    json.dump(report, open(a.out, 'w'), indent=1)
