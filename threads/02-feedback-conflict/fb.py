"""H2 feedback-conflict testbed: nested scalar 2->4 bit codes under LDLQ feedback.

Conventions (row-vector): W [m, n], proxy loss tr(E H E^T), E = What - W.
H = M D M^T with M unit upper triangular  =>  loss = sum_j D_j ||E_j + sum_{k<j} E_k M_kj||^2,
so LDLQ target for column j is t_j = w_j - sum_{k<j} E_k M_kj and the greedy is exact:
loss = sum_j D_j ||q_j - t_j||^2 .
Nested code: 4-bit index i in 0..15, base (2-bit) index = i >> 2 (contiguous cells).
2-bit decoder c2[4], 4-bit decoder c4[16] (4-bit decoder freely reinterprets base bits).
"""
import math
import torch

LLOYD4 = [-1.5104, -0.4528, 0.4528, 1.5104]
_L16 = [0.1284, 0.3881, 0.6568, 0.9424, 1.2562, 1.6180, 2.0690, 2.7326]
LLOYD16 = [-v for v in _L16[::-1]] + _L16


def rand_orth(n, seed, device):
    g = torch.Generator(device='cpu').manual_seed(seed)
    a = torch.randn(n, n, generator=g, dtype=torch.float64)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diagonal(r))[None]
    return q.to(device)


def udu(H, damp=1e-3):
    """H = M D M^T, M unit upper. Returns M (fp32), D (fp64)."""
    n = H.shape[0]
    H = H.double().clone()
    H += damp * H.diagonal().mean() * torch.eye(n, device=H.device, dtype=H.dtype)
    Hf = H.flip(0).flip(1)
    C = torch.linalg.cholesky(Hf)
    U = C.flip(0).flip(1)          # upper, H = U U^T
    d = U.diagonal()
    M = U / d[None, :]
    return M.float().contiguous(), d.square()


class Problem:
    """A rotated, row-normalised weight with its Hessian."""

    def __init__(self, W, H, seed=0, rotate=True, damp=1e-3):
        dev = W.device
        m, n = W.shape
        self.H = H.double()
        if rotate:
            self.Qi = rand_orth(n, seed, dev)
            self.Qo = rand_orth(m, seed + 1, dev)
            Wr = self.Qo.T @ W.double() @ self.Qi
            Hr = self.Qi.T @ self.H @ self.Qi
        else:
            self.Qi = self.Qo = None
            Wr, Hr = W.double(), self.H
        self.Hr = Hr
        self.s = Wr.square().mean(1).sqrt()          # per-row scale
        self.Wn = (Wr / self.s[:, None]).float().contiguous()
        self.M, self.D = udu(Hr, damp)
        self.den = float(torch.einsum('ij,jk,ik->', Wr, Hr, Wr))
        self.Wr = Wr

    def loss(self, Qn):
        """relative proxy tr(E H E^T)/tr(W H W^T) with the undamped rotated H."""
        E = (Qn.double() - self.Wn.double()) * self.s[:, None]
        return float(torch.einsum('ij,jk,ik->', E, self.Hr, E)) / self.den

    def dequant(self, Qn):
        Wr = Qn.double() * self.s[:, None]
        if self.Qi is None:
            return Wr.float()
        return (self.Qo @ Wr @ self.Qi.T).float()


@torch.no_grad()
def fit(P, c2, c4, rule, lam=0.0, block=128, mu=0.5):
    """One LDLQ pass. rule in nat2 | nat4 | seq | blend | joint | joint3.
    Returns (Q2n, Q4n, idx, T2, T4) with targets actually used by each view."""
    W, M = P.Wn, P.M
    m, n = W.shape
    dev = W.device
    c2 = c2.to(dev).float(); c4 = c4.to(dev).float()
    child = torch.arange(16, device=dev).view(4, 4)
    base_of = torch.arange(16, device=dev) >> 2
    c2of = c2[base_of]                                  # [16]
    E2 = torch.zeros(m, n, device=dev); E4 = torch.zeros(m, n, device=dev)
    T2all = torch.empty(m, n, device=dev); T4all = torch.empty(m, n, device=dev)
    idx = torch.empty(m, n, dtype=torch.uint8, device=dev)
    for a in range(0, n, block):
        b = min(n, a + block)
        T2 = W[:, a:b] - (E2[:, :a] @ M[:a, a:b] if a else 0)
        T4 = W[:, a:b] - (E4[:, :a] @ M[:a, a:b] if a else 0)
        T2 = T2.clone(); T4 = T4.clone()
        Mb = M[a:b, a:b]
        for jj in range(b - a):
            t2 = T2[:, jj]; t4 = T4[:, jj]
            if rule == 'nat2' or rule == 'seq' or rule == 'blend':
                t = t2 if rule != 'blend' else (1 - lam) * t2 + lam * t4
                bi = (t[:, None] - c2[None]).abs().argmin(1)
                ch = child[bi]                                   # [m,4]
                k = (t4[:, None] - c4[ch]).abs().argmin(1)
                i = ch.gather(1, k[:, None]).squeeze(1)
            elif rule == 'nat4':
                i = (t4[:, None] - c4[None]).abs().argmin(1)
            elif rule == 'shared':   # MatGPTQ: one state, lam-weighted selection, mu-weighted propagated error
                cost = (1 - lam) * (t2[:, None] - c2of[None]).square() + lam * (t2[:, None] - c4[None]).square()
                i = cost.argmin(1)
            elif rule == 'joint':
                cost = (1 - lam) * (t2[:, None] - c2of[None]).square() + lam * (t4[:, None] - c4[None]).square()
                i = cost.argmin(1)
            else:
                raise ValueError(rule)
            q2 = c2[i >> 2]; q4 = c4[i]
            w = W[:, a + jj]
            e2 = q2 - w; e4 = q4 - w
            if rule == 'shared':
                e2 = e4 = (1 - mu) * e2 + mu * e4
            E2[:, a + jj] = e2; E4[:, a + jj] = e4
            T2all[:, a + jj] = t2; T4all[:, a + jj] = t4 if rule != 'shared' else t2
            idx[:, a + jj] = i.to(torch.uint8)
            if jj + 1 < b - a:
                T2[:, jj + 1:] -= e2[:, None] * Mb[jj, jj + 1:][None]
                T4[:, jj + 1:] -= e4[:, None] * Mb[jj, jj + 1:][None]
    Q2 = c2[(idx.long() >> 2)]; Q4 = c4[idx.long()]
    return Q2, Q4, idx, T2all, T4all


def refit_books(P, idx, T2, T4):
    """Decoder refit: weighted centroid of the targets each view was steering to."""
    wgt = (P.D.float()[None, :] * P.s.float()[:, None].square())
    i = idx.long()
    c4 = torch.zeros(16, device=T4.device, dtype=torch.float64); n4 = torch.zeros_like(c4)
    c4.index_add_(0, i.flatten(), (wgt * T4).flatten().double()); n4.index_add_(0, i.flatten(), wgt.flatten().double())
    b = i >> 2
    c2 = torch.zeros(4, device=T4.device, dtype=torch.float64); n2 = torch.zeros_like(c2)
    c2.index_add_(0, b.flatten(), (wgt * T2).flatten().double()); n2.index_add_(0, b.flatten(), wgt.flatten().double())
    return (c2 / n2.clamp_min(1e-30)).float(), (c4 / n4.clamp_min(1e-30)).float(), n2, n4


def run(P, rule, lam=0.0, passes=3, c2=None, c4=None, refit2=True, refit4=True, mu=0.5):
    c2 = torch.tensor(LLOYD4) if c2 is None else c2
    c4 = torch.tensor(LLOYD16) if c4 is None else c4
    hist = []
    for p in range(passes):
        Q2, Q4, idx, T2, T4 = fit(P, c2, c4, rule, lam, mu=mu)
        hist.append((P.loss(Q2), P.loss(Q4)))
        if p + 1 < passes:
            n2c, n4c, n2, n4 = refit_books(P, idx, T2, T4)
            if refit2: c2 = torch.where(n2 > 0, n2c, c2.to(n2c.device))
            if refit4: c4 = torch.where(n4 > 0, n4c, c4.to(n4c.device))
    return dict(l2=hist[-1][0], l4=hist[-1][1], hist=hist, c2=c2.cpu().tolist(), c4=c4.cpu().tolist(), Q2=Q2, Q4=Q4, idx=idx)


@torch.no_grad()
def innov_refine(P, c2, c4=None, passes=3, G=None):
    """Base = native LDLQ-2. Refinement quantizes the base's own target T2 inside the base cell
    (successive refinement of the innovation). 4-bit decoder: W4 = Q2 + (Q4' - Q2) @ G,
    G = M^{-1} is exact (loss4 = sum_j D_j ||Q4'_j - T2_j||^2)."""
    c4 = torch.tensor(LLOYD16) if c4 is None else c4
    Q2, _, idx, T2, _ = fit(P, c2, c4, 'nat2')
    b = (idx.long() >> 2)
    dev = T2.device
    c4 = c4.to(dev).float()
    wgt = (P.D.float()[None, :] * P.s.float()[:, None].square())
    for p in range(passes):
        cand = c4.view(4, 4)[b]                                  # [m,n,4]
        k = (T2[..., None] - cand).abs().argmin(-1)
        i = b * 4 + k
        num = torch.zeros(16, device=dev, dtype=torch.float64).index_add_(0, i.flatten(), (wgt * T2).flatten().double())
        den = torch.zeros(16, device=dev, dtype=torch.float64).index_add_(0, i.flatten(), wgt.flatten().double())
        c4 = torch.where(den > 0, num / den.clamp_min(1e-30), c4.double()).float()
    Q4p = c4[i]
    delta = Q4p - Q2
    if G is None:
        corr = torch.linalg.solve_triangular(P.M, delta, upper=True, left=False)
    else:
        corr = delta @ G
    W4 = Q2 + corr
    return dict(l2=P.loss(Q2), l4=P.loss(W4), l4_nocorr=P.loss(Q4p), Q2=Q2, Q4=W4, delta=delta, idx=i, c4=c4.cpu().tolist())


@torch.no_grad()
def cd_joint(P, idx, c2, c4, a, b, sweeps=2):
    """Coordinate descent on a*L2 + b*L4 (exact quadratic forms, undamped rotated H).
    Each weight picks among all 16 nested indices given the rest."""
    dev = P.Wn.device
    H = P.Hr.float()
    c2 = torch.as_tensor(c2, device=dev).float(); c4 = torch.as_tensor(c4, device=dev).float()
    c2of = c2[torch.arange(16, device=dev) >> 2]
    i = idx.long().clone()
    E2 = c2[i >> 2] - P.Wn; E4 = c4[i] - P.Wn
    G2 = E2 @ H; G4 = E4 @ H
    n = H.shape[0]
    for s in range(sweeps):
        changed = 0
        for j in range(n):
            h = H[j, j]
            d2 = c2of[None] - c2[i[:, j] >> 2][:, None]          # [m,16]
            d4 = c4[None] - c4[i[:, j]][:, None]
            dj = a * (2 * d2 * G2[:, j:j+1] + d2.square() * h) + b * (2 * d4 * G4[:, j:j+1] + d4.square() * h)
            new = dj.argmin(1)
            mv = new != i[:, j]
            if mv.any():
                changed += int(mv.sum())
                del2 = c2of[new] - c2[i[:, j] >> 2]; del4 = c4[new] - c4[i[:, j]]
                G2 += del2[:, None] * H[j][None]; G4 += del4[:, None] * H[j][None]
                i[:, j] = new
        # print('cd sweep', s, 'changed', changed)
    return c2[i >> 2], c4[i], i
