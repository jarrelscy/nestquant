"""Trellis variant (EXL3 mul1 bitshift trellis, CUDA Viterbi) of the H2 testbed.
Block-LDL with 16-column blocks (as EXL3); tiles = 256 consecutive output rows of one input column."""
import sys, torch
sys.path.insert(0, '/home/coder/git/nestquant/threads/05-exl3-harness')
import harness as h
from fb import Problem
TQ = h.ExtTileQuantizer('mul1')
GAIN = {2: 1.0, 4: 0.9}


def block_udu(P, B=16):
    Ms, D = P.M.double(), P.D
    U = Ms * D.sqrt()[None]
    n = U.shape[0]
    Mb = torch.empty_like(U); Db = []
    for a in range(0, n, B):
        Ubb = U[a:a+B, a:a+B]
        inv = torch.linalg.solve_triangular(Ubb, torch.eye(B, device=U.device, dtype=U.dtype), upper=True)
        Mb[:, a:a+B] = U[:, a:a+B] @ inv
        Db.append(Ubb @ Ubb.T)
    return Mb.float(), torch.stack(Db).float()


def tq(x, K, gain):
    """x [m, B] -> trellis-quantised along m (tiles of 256), per column of the block."""
    m, B = x.shape
    t = (x.T * gain).reshape(-1, 256).contiguous()
    q, _ = TQ(t, K)
    return (q.float().reshape(B, m).T) / gain


@torch.no_grad()
def fit_trellis(P, rule, lam=0.0, B=16, gam=1.0):
    if not hasattr(P, 'Mb'):
        P.Mb, P.Db = block_udu(P, B)
    W, M = P.Wn, P.Mb
    m, n = W.shape
    E2 = torch.zeros_like(W); E4 = torch.zeros_like(W)
    Q2 = torch.zeros_like(W); Q4 = torch.zeros_like(W); T2all = torch.zeros_like(W); Dl = torch.zeros_like(W)
    for a in range(0, n, B):
        b = a + B
        t2 = W[:, a:b] - gam * (E2[:, :a] @ M[:a, a:b] if a else 0)
        t4 = W[:, a:b] - (E4[:, :a] @ M[:a, a:b] if a else 0)
        w = W[:, a:b]
        if rule == 'nat4':
            q4 = tq(t4, 4, GAIN[4]); q2 = torch.zeros_like(q4) + float('nan')
            E4[:, a:b] = q4 - w; E2[:, a:b] = 0
        else:
            tb = t2 if rule in ('nat2', 'seq', 'innov') else (1 - lam) * t2 + lam * t4
            q2 = tq(tb, 2, GAIN[2])
            if rule == 'innov':
                r = t2 - q2
            else:
                r = t4 - q2
            rs = r.square().mean().sqrt().clamp_min(1e-8)          # per-block residual scale (side info)
            d = tq(r, 2, GAIN[2] / rs)
            q4 = q2 + d
            Dl[:, a:b] = d
            E2[:, a:b] = q2 - w; E4[:, a:b] = q4 - w
        Q2[:, a:b] = q2; Q4[:, a:b] = q4
    out = dict(Q2=Q2, Q4=Q4)
    if rule == 'innov':
        out['Q4'] = Q2 + torch.linalg.solve_triangular(M, Dl, upper=True, left=False)
        out['Q4_nocorr'] = Q4
    return out


def run_trellis(P, rule, lam=0.0, gam=1.0):
    o = fit_trellis(P, rule, lam, gam=gam)
    l2 = P.loss(o['Q2']) if rule != 'nat4' else float('nan')
    return dict(l2=l2, l4=P.loss(o['Q4']), Q2=o['Q2'], Q4=o['Q4'])


@torch.no_grad()
def g_refine(P, Q2, G):
    """Refinement through a decoder-side linear map G (applied to activations, z = G x):
    W4 = Q2 + Delta @ G, Delta = LDLQ-trellis-2 of target -E2 G^{-1} under Hessian G H G^T.
    G = I is 'seq', G = M^{-1} is 'innov' (exact)."""
    E2 = Q2 - P.Wn
    n = G.shape[0]
    target = -torch.linalg.solve(G.double().T, E2.double().T).T.float()     # -E2 G^{-1}
    HG = (G.double() @ P.Hr @ G.double().T)
    P2 = Problem(target * P.s.float()[:, None], HG, rotate=False)
    o = fit_trellis(P2, 'nat2')
    Delta = (o['Q2'].double() * P2.s[:, None] / P.s[:, None]).float()
    return Q2 + Delta @ G


@torch.no_grad()
def fit_trellis_sel(P, mu, lams=(0.0, 0.25, 0.5, 0.75, 1.0), L2=1.0, L4=1.0, B=16):
    """Per-16-column-block choice of the blend weight: pick lam minimising
    (1-mu)*blockloss2/L2 + mu*blockloss4/L4 (exact block LDL costs, both feedback states)."""
    if not hasattr(P, 'Mb'):
        P.Mb, P.Db = block_udu(P, B)
    W, M = P.Wn, P.Mb
    m, n = W.shape
    s2 = P.s.float()[:, None].square()
    E2 = torch.zeros_like(W); E4 = torch.zeros_like(W); Q2 = torch.zeros_like(W); Q4 = torch.zeros_like(W)
    chosen = []
    for a in range(0, n, B):
        b = a + B; Db = P.Db[a // B]
        t2 = W[:, a:b] - (E2[:, :a] @ M[:a, a:b] if a else 0)
        t4 = W[:, a:b] - (E4[:, :a] @ M[:a, a:b] if a else 0)
        best = None
        for lam in lams:
            q2 = tq((1 - lam) * t2 + lam * t4, 2, GAIN[2])
            r = t4 - q2; rs = r.square().mean().sqrt().clamp_min(1e-8)
            q4 = q2 + tq(r, 2, GAIN[2] / rs)
            e2 = q2 - t2; e4 = q4 - t4
            c2 = float(((e2 @ Db) * e2 * s2).sum()) / P.den; c4 = float(((e4 @ Db) * e4 * s2).sum()) / P.den
            c = (1 - mu) * c2 / L2 + mu * c4 / L4
            if best is None or c < best[0]: best = (c, lam, q2, q4)
        _, lam, q2, q4 = best; chosen.append(lam)
        w = W[:, a:b]; E2[:, a:b] = q2 - w; E4[:, a:b] = q4 - w; Q2[:, a:b] = q2; Q4[:, a:b] = q4
    return dict(l2=P.loss(Q2), l4=P.loss(Q4), Q2=Q2, Q4=Q4, lams=chosen)


@torch.no_grad()
def fit_trellis_alt(P, lam, iters=3, B=16):
    """Refinement-aware blend: base Viterbi target (1-lam)*t2 + lam*(t4 - d), where d is the refinement's
    current reconstructed delta (alternating base/refinement per 16-col block, iters rounds)."""
    if not hasattr(P, 'Mb'):
        P.Mb, P.Db = block_udu(P, B)
    W, M = P.Wn, P.Mb
    m, n = W.shape
    E2 = torch.zeros_like(W); E4 = torch.zeros_like(W); Q2 = torch.zeros_like(W); Q4 = torch.zeros_like(W)
    for a in range(0, n, B):
        b = a + B
        t2 = W[:, a:b] - (E2[:, :a] @ M[:a, a:b] if a else 0)
        t4 = W[:, a:b] - (E4[:, :a] @ M[:a, a:b] if a else 0)
        d = torch.zeros_like(t2)
        for it in range(iters):
            q2 = tq((1 - lam) * t2 + lam * (t4 - d), 2, GAIN[2])
            r = t4 - q2; rs = r.square().mean().sqrt().clamp_min(1e-8)
            d = tq(r, 2, GAIN[2] / rs)
        q4 = q2 + d
        w = W[:, a:b]; E2[:, a:b] = q2 - w; E4[:, a:b] = q4 - w; Q2[:, a:b] = q2; Q4[:, a:b] = q4
    return dict(l2=P.loss(Q2), l4=P.loss(Q4), Q2=Q2, Q4=Q4)
