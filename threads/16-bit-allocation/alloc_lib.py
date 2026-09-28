"""Thread 16: fixed-budget bit allocation for the nested 2+2 mul1 code (thread 02 testbed generalised).
Per 16-column LDL block b: base rate Kb[b] (EXL3 mul1 bitshift trellis, L16) and refinement rate Kr[b]
(0 = base only; second mul1 trellis on the rotated residual, per-block RMS scale delta).  Base target is the
thread-02 blend (1-lam) t2 + lam t4 with separate E2/E4 feedback states.  Optional per-block cost curves
(exact block-LDL greedy costs) for benefit-per-byte allocation."""
import os, sys, json, math, torch, torch.nn.functional as F
T02 = '/home/coder/git/nestquant/threads/02-feedback-conflict'; T05 = '/home/coder/git/nestquant/threads/05-exl3-harness'
for p in ('/home/coder/git/orbit-duet', T02, T05):
    if p not in sys.path: sys.path.insert(0, p)
import harness as hh; hh.gpu_cap(12)
from fb import Problem
TQ = hh.ExtTileQuantizer('mul1')
GAIN = {1: 1.1, 1.5: 1.05, 2: 1.0, 2.5: 0.95, 3: 0.95, 3.5: 0.92, 4: 0.9}   # MSE-optimal on N(0,1) (gaincal.py); 2/4 = thread 02
SCR = '/tmp/nestquant/16-bit-allocation'
RUN = '/home/coder/git/orbit-duet/runs/glm53_pilot_matched_l{L}/statistics/l{L}_e{E}_training_sample.pt'
DAMP = (0.5, 0.5, 1.0)
PROJ = ['gate', 'up', 'down']

def K_(k):
    return int(k) if float(k).is_integer() else float(k)

def tq(x, K, gain=None):
    """x [m, B] -> trellis-quantised along m (tiles of 256 rows of one column).  gain scalar or None (GAIN[K])."""
    if K == 0: return torch.zeros_like(x)
    K = K_(K); g = GAIN[K] if gain is None else gain
    m, B = x.shape
    t = (x.T * g).reshape(-1, 256).contiguous()
    q, _ = TQ(t, K)
    return (q.float().reshape(B, m).T) / g

def block_udu(P, B=16):
    from fbt import block_udu as bu
    return bu(P, B)

def load(L, E, a=0.75):
    """Teacher weights + thread-08 H (0.75 uniform-token + 0.25 p^2-routed, each /trace) from the training sample."""
    data = hh.load_expert(L, E)
    g, u, d = data.teacher
    ts = torch.load(RUN.format(L=L, E=E), map_location='cpu', mmap=True, weights_only=False)
    n = len(ts['x'])
    def grams(pw):
        Hx = torch.zeros(6144, 6144, device='cuda'); Ha = torch.zeros(2048, 2048, device='cuda')
        for i in range(0, n, 2048):
            x = ts['x'][i:i+2048].cuda().float(); r = ts['p'][i:i+2048].cuda().float().pow(pw / 2)[:, None]
            act = F.silu(x @ g.T) * (x @ u.T)
            Hx.addmm_((x*r).T, x*r); Ha.addmm_((act*r).T, act*r)
        return [Hx, Ha]
    A = grams(2); U = grams(0)
    Hs = [(1 - a) * A[k] / A[k].diagonal().mean() + a * U[k] / U[k].diagonal().mean() for k in range(2)]
    del A, U, ts; torch.cuda.empty_cache()
    return data, [Hs[0], Hs[0], Hs[1]]

def problem(W, H, mi):
    P = Problem(W, H, damp=DAMP[mi])
    P.Mb, P.Db = block_udu(P)
    P.s2 = P.s.float()[:, None].square()
    return P

def bcost(P, e, b):
    return float(((e @ P.Db[b]) * e * P.s2).sum()) / P.den

@torch.no_grad()
def fit(P, lam, Kb, Kr, cand_b=(), cand_r=(), B=16):
    """Kb, Kr: per-block rates (len n/B).  cand_*: rates for which per-block cost curves are recorded
    (given the actual running feedback targets of this fit)."""
    W, M = P.Wn, P.Mb
    m, n = W.shape; nb = n // B
    E2 = torch.zeros_like(W); E4 = torch.zeros_like(W); Q2 = torch.zeros_like(W); Q4 = torch.zeros_like(W)
    c2 = [dict() for _ in range(nb)]; c4 = [dict() for _ in range(nb)]
    for bi in range(nb):
        a = bi * B; b = a + B
        t2 = W[:, a:b] - (E2[:, :a] @ M[:a, a:b] if a else 0)
        t4 = W[:, a:b] - (E4[:, :a] @ M[:a, a:b] if a else 0)
        tb = (1 - lam) * t2 + lam * t4
        for K in cand_b:
            c2[bi][K] = bcost(P, tq(tb, K) - t2, bi)
        q2 = tq(tb, Kb[bi])
        r = t4 - q2; rs = r.square().mean().sqrt().clamp_min(1e-8)
        for K in cand_r:
            c4[bi][K] = bcost(P, q2 + rs * tq(r / rs, K) - t4, bi)
        q4 = q2 + rs * tq(r / rs, Kr[bi])
        w = W[:, a:b]
        E2[:, a:b] = q2 - w; E4[:, a:b] = q4 - w; Q2[:, a:b] = q2; Q4[:, a:b] = q4
    return dict(Q2=Q2, Q4=Q4, l2=P.loss(Q2), l4=P.loss(Q4), c2=c2, c4=c4)

def bits(m, n, Kb, Kr, B=16, tp=8, down=False, mapbits=2):
    """Total stored bits for one projection: trellis planes + per-block delta (fp16, replicated per TP shard for
    gate/up) + fp16 per-row and per-column scales (like EXL3 suh/svh) + a per-block rate map per plane."""
    tb = sum(Kb) * B * m + sum(Kr) * B * m
    nref = sum(1 for k in Kr if k > 0)
    dbits = 16 * nref * (1 if down else tp)
    return tb + dbits + 16 * (m + n) + mapbits * 2 * len(Kb)

def save_w(P, o, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(dict(w2=P.dequant(o['Q2']).bfloat16().cpu(), w4=P.dequant(o['Q4']).bfloat16().cpu()), path)
