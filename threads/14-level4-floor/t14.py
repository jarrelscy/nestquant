"""Thread 14 testbed: nested 2+R mul1 trellis on the thread-02 fbt pipeline (random orthogonal rotations,
per-row RMS, 16-column block LDL, separate E2/E4 feedback), generalised to
  - per-projection / per-block base and residual rates (K lists, half-bit via exllamav3 frac kernel),
  - residual gain / scale variants, conditional-residual diagnostics.
H = thread-08: 0.25 H_routed(p^2)/tr + 0.75 H_uniform/tr, damp 0.5/0.5/1.0 (Problem damp = sigma*mean diag)."""
import os, sys, math, torch
import torch.nn.functional as F
sys.path.insert(0, '/home/coder/git/nestquant/threads/05-exl3-harness')
sys.path.insert(0, '/home/coder/git/nestquant/threads/02-feedback-conflict')
sys.path.insert(0, '/home/coder/git/orbit-duet')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as h
h.gpu_cap(12)
from fb import Problem
from fbt import block_udu
TQ = h.ExtTileQuantizer('mul1')
SCR = '/tmp/nestquant/14-level4-floor'
os.makedirs(SCR, exist_ok=True)
DAMP = (0.5, 0.5, 1.0)
PROJ = ['gate', 'up', 'down']


def stats_dir(L):
    return f'/home/coder/git/orbit-duet/runs/glm53_pilot_matched_l{L}/statistics'


@torch.no_grad()
def hessians(L, E, Ws, a=0.75):
    f = f'{SCR}/H_l{L}_e{E}.pt'
    if os.path.exists(f):
        return [x.cuda() for x in torch.load(f)]
    ts = torch.load(f'{stats_dir(L)}/l{L}_e{E}_training_sample.pt', map_location='cpu', mmap=True, weights_only=False)
    g, u = Ws[0].cuda().float(), Ws[1].cuda().float()
    def grams(pw):
        Hx = torch.zeros(6144, 6144, device='cuda'); Ha = torch.zeros(2048, 2048, device='cuda')
        n = len(ts['x'])
        for i in range(0, n, 2048):
            x = ts['x'][i:i+2048].cuda().float(); r = ts['p'][i:i+2048].cuda().float().pow(pw / 2)[:, None]
            act = F.silu(x @ g.T) * (x @ u.T)
            Hx.addmm_((x * r).T, x * r); Ha.addmm_((act * r).T, act * r)
        return [Hx, Hx, Ha]
    A = grams(2); U = grams(0)
    Hs = [(1 - a) * A[k] / A[k].diagonal().mean() + a * U[k] / U[k].diagonal().mean() for k in range(3)]
    torch.save([x.cpu() for x in Hs], f)
    return Hs


def tq(x, K, gain):
    """x [m, B] -> trellis-quantised along m (tiles of 256), per column. Returns (values, state idx)."""
    m, B = x.shape
    t = (x.T * gain).reshape(-1, 256).contiguous()
    K = int(K) if float(K).is_integer() else float(K)
    if (2 * K) % 1:                      # not a whole/half bit: pattern-rate torch Viterbi (patvit.py)
        import patvit
        q, idx = patvit.quantize_tiles(t, K)
    else:
        q, idx = TQ(t, K)
    return q.float().reshape(B, m).T / gain, idx.reshape(B, m).T


def problem(W, H, mi):
    return Problem(W.cuda().float(), H, damp=DAMP[mi])


def _k(K, j):
    return K[j] if isinstance(K, (list, tuple)) else K


@torch.no_grad()
def fit(P, lam=0.3, Kb=2, Kr=2, B=16, gb=1.0, gr=1.0, resid_fn=None, collect=False, nat=None):
    """Nested fit. Kb/Kr: scalar or per-16-col-block list (Kr=0 -> no residual in that block).
    gb/gr: trellis gain for base / residual (target scaled to unit RMS * g before Viterbi).
    resid_fn(r, q2, idx2, j) -> d  optional custom residual quantiser (r in normalised units).
    nat=K : native single-stage code at rate K (for anchors)."""
    if not hasattr(P, 'Mb'):
        P.Mb, P.Db = block_udu(P, B)
    W, M = P.Wn, P.Mb
    m, n = W.shape
    E2 = torch.zeros_like(W); E4 = torch.zeros_like(W)
    Q2 = torch.zeros_like(W); Q4 = torch.zeros_like(W)
    col = [] if collect else None
    for j, a in enumerate(range(0, n, B)):
        b = a + B
        w = W[:, a:b]
        t2 = w - (E2[:, :a] @ M[:a, a:b] if a else 0)
        t4 = w - (E4[:, :a] @ M[:a, a:b] if a else 0)
        if nat is not None:
            q4, _ = tq(t4, _k(nat, j), gb)
            Q4[:, a:b] = q4; E4[:, a:b] = q4 - w
            continue
        kb = _k(Kb, j)
        tb = (1 - lam) * t2 + lam * t4
        q2, i2 = tq(tb, kb, gb)
        r = t4 - q2
        kr = _k(Kr, j)
        if kr == 0:
            d = torch.zeros_like(r)
        else:
            rs = r.square().mean().sqrt().clamp_min(1e-8)
            if resid_fn is not None:
                d = resid_fn(r / rs, q2, i2, j, kr) * rs
            else:
                d, _ = tq(r / rs, kr, gr); d = d * rs
        if collect:
            col.append(dict(r=(r / r.square().mean().sqrt()).cpu(), q2=q2.cpu(), i2=i2.cpu(), tb=tb.cpu()))
        q4 = q2 + d
        E2[:, a:b] = q2 - w; E4[:, a:b] = q4 - w
        Q2[:, a:b] = q2; Q4[:, a:b] = q4
    out = dict(Q2=Q2, Q4=Q4, l2=P.loss(Q2), l4=P.loss(Q4))
    if collect:
        out['col'] = col
    return out


_DATA = {}


def data(L, E):
    if (L, E) not in _DATA:
        _DATA.clear(); torch.cuda.empty_cache()
        _DATA[(L, E)] = h.load_expert(L, E)
    return _DATA[(L, E)]


def evaluate(L, E, methods):
    """methods: {name: [gate, up, down] dequantised}. Returns compact table (routed/forced/ood)."""
    tab = h.table(h.evaluate(data(L, E), methods))
    return {k: dict(routed=v['all/routed'], forced=v['all/forced'], ood=v['ood/forced'], ood_routed=v.get('ood/routed')) for k, v in tab.items()}
