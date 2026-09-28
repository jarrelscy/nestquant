"""MatGPTQ-style nested (Matryoshka) 2/4-bit GPTQ on GLM experts, via the thread-05 harness.

One 4-bit code c per weight; the 2-bit view is k = c >> 2 (the two MSBs).
Grids (per output row, per input-column group of size g):
  'centre' : uniform asymmetric 4-bit cells  v4(c) = lo + s*(c+.5); 2-bit = centre of its 4 sub-cells
             v2(k) = lo + 4s*(k+.5)  (one (lo, s) pair = 32 bits/group, shared by both views)
  'msb'    : MatQuant/MatGPTQ integer grid v4(c) = lo + s*c, v2(k) = lo + 4s*k (plain MSB truncation, biased)
  'sep'    : separate per-group 2-bit (lo2, S2) and 4-bit (lo4, s4) uniform grids (64 bits/group at 4 bit)
  'nu'     : non-uniform nested Gaussian Lloyd grid (2-bit Lloyd centroids, each cell split by conditional Lloyd),
             symmetric, one fp16 scale per group (16 bits/group)
Error feedback (GPTQ over input columns, H damped exactly like EXL3: H/count + 0.03*mean(diag) I):
  'dual'   : separate 2-bit and 4-bit feedback targets W2, W4; code = argmin l2 (w2-v2)^2 + l4 (w4-v4)^2
             (each precision gets its own GPTQ compensation = cross-bit error feedback through the shared code)
  'shared' : one target, fed back the l-weighted mean error (MatGPTQ feeds the unweighted mean)
Optional incoherence: random signs + 128-block Hadamard on both sides (as EXL3).
"""
import math, torch
import harness as hz

LL2 = None


def lloyd_tables():
    """Nested Gaussian Lloyd grid: T2[4], T4[16] with T4[4k:4k+4] splitting 2-bit cell k (T4 sub-cells are
    themselves the 2-cell conditional Lloyd of each of the two halves -> 4 per 2-bit cell)."""
    global LL2
    if LL2 is not None:
        return LL2
    g = torch.Generator().manual_seed(0)
    x = torch.randn(4_000_000, generator=g, dtype=torch.float64).sort().values

    def lloyd(xs, n, it=200):
        c = torch.quantile(xs[:: max(1, len(xs) // 100000)], torch.linspace(0.5 / n, 1 - 0.5 / n, n, dtype=torch.float64))
        for _ in range(it):
            t = (c[1:] + c[:-1]) / 2
            a = torch.bucketize(xs, t)
            c = torch.stack([xs[a == i].mean() for i in range(n)])
        return c, torch.bucketize(xs, (c[1:] + c[:-1]) / 2)
    c2, a2 = lloyd(x, 4)
    t4 = []
    for k in range(4):
        sub = x[a2 == k]
        c, _ = lloyd(sub, 4, 100)       # 4 sub-centroids inside cell k (not re-centred: 2-bit keeps c2[k])
        t4.append(c)
    T4 = torch.cat(t4)
    LL2 = (c2.float(), T4.float())
    return LL2


def had_blk(n, dev):
    return hz._had(128).to(dev)     # [128,128] orthonormal


def rot_in(W, su, Hb):          # W @ diag(su) @ blkdiag(Hb)^T
    r, c = W.shape
    return ((W * su[None]).reshape(r, c // 128, 128) @ Hb.T).reshape(r, c)


def unrot_in(W, su, Hb):
    r, c = W.shape
    return (W.reshape(r, c // 128, 128) @ Hb).reshape(r, c) * su[None]


def rot_out(W, sv, Hb):         # blkdiag(Hb) @ diag(sv) @ W
    return rot_in(W.T, sv, Hb).T


def unrot_out(W, sv, Hb):
    return unrot_in(W.T, sv, Hb).T


class Grid:
    """Per-row params for one group; values(): v2 [r,16], v4 [r,16] (v2 indexed by code c -> c>>2)."""

    def __init__(self, kind):
        self.kind = kind

    def values(self, p):
        c = torch.arange(16, device=p[0].device, dtype=torch.float32)
        k = torch.div(c, 4, rounding_mode="floor")
        if self.kind == "centre":
            lo, s = p
            return lo + 4 * s * (k + .5), lo + s * (c + .5)
        if self.kind == "msb":
            lo, s = p
            return lo + 4 * s * k, lo + s * c
        if self.kind == "sep":
            lo2, S2, lo4, s4 = p
            return lo2 + S2 * (k + .5), lo4 + s4 * (c + .5)
        if self.kind == "nu":
            (s,) = p
            T2, T4 = lloyd_tables()
            T2 = T2.to(s.device); T4 = T4.to(s.device)
            return s * T2[k.long()][None], s * T4[None]
        raise ValueError(self.kind)

    def bits(self, level):
        per = {"centre": (32, 32), "msb": (32, 32), "sep": (32, 64), "nu": (16, 16)}[self.kind]
        return per[0] if level == 2 else per[1]


def _codes(x2, x4, v2, v4, l2, l4):
    # x*: [r, g], v*: [r, 16] -> cost [r, g, 16]
    cost = l2 * (x2[..., None] - v2[:, None]).square() + l4 * (x4[..., None] - v4[:, None]).square()
    return cost.argmin(-1)


def find_params(kind, x2, x4, l2, l4, nshrink=24):
    """Grid search per row (shrink factor of min/max range, or scale multiple of rms for 'nu')."""
    grid = Grid(kind)
    r = x2.shape[0]
    best = None; bestc = torch.full((r,), float("inf"), device=x2.device)
    ps = torch.linspace(0.3, 1.0, nshrink).tolist()

    def evalp(p):
        v2, v4 = grid.values(_col(p))
        c = _codes(x2, x4, v2, v4, l2, l4)
        e2 = (x2 - v2.gather(1, c.reshape(r, -1)).reshape_as(x2)).square().sum(1)
        e4 = (x4 - v4.gather(1, c.reshape(r, -1)).reshape_as(x4)).square().sum(1)
        return l2 * e2 + l4 * e4

    if kind in ("centre", "msb"):
        xr = x4 if l4 > 0 else x2
        mn, mx = xr.min(1).values, xr.max(1).values
        # joint objective may want the 2-bit range: also search on x2 range
        cands = []
        for p in ps:
            for src in ([(mn, mx)] + ([(x2.min(1).values, x2.max(1).values)] if (l2 > 0 and l4 > 0) else [])):
                a, b = src
                mid = (a + b) / 2; half = (b - a) / 2 * p
                lo = mid - half; span = 2 * half
                s = span / (16 if kind == "centre" else 15)
                s = torch.clamp(s, min=1e-8)
                cands.append((lo, s))
        for p in cands:
            c = evalp(p)
            upd = c < bestc
            bestc = torch.where(upd, c, bestc)
            best = p if best is None else tuple(torch.where(upd, a, b) for a, b in zip(p, best))
        return best
    if kind == "sep":
        # fit the two grids independently on their own targets (2-bit MSE, 4-bit MSE), then keep
        p2 = find_params("centre", x2, x2, 1.0, 0.0, nshrink)            # centre-2 grid: lo + 4s(k+.5)
        lo2, s2 = p2; S2 = 4 * s2
        # 4-bit params searched with the joint objective given the 2-bit grid
        mn, mx = x4.min(1).values, x4.max(1).values
        for p in ps:
            mid = (mn + mx) / 2; half = (mx - mn) / 2 * p
            q = (lo2, S2, mid - half, torch.clamp(2 * half / 16, min=1e-8))
            c = evalp(q)
            upd = c < bestc
            bestc = torch.where(upd, c, bestc)
            best = q if best is None else tuple(torch.where(upd, a, b) for a, b in zip(q, best))
        return best
    if kind == "nu":
        rms = torch.sqrt(((l2 * x2.square() + l4 * x4.square()) / (l2 + l4)).mean(1)).clamp(min=1e-8)
        for m in torch.linspace(0.6, 1.4, 33).tolist():
            q = (rms * m,)
            q = (q[0][:, None],)
            c = evalp(q)
            upd = c < bestc
            bestc = torch.where(upd, c, bestc)
            best = q if best is None else (torch.where(upd[:, None], q[0], best[0]),)
        return best
    raise ValueError(kind)


def _col(p):
    return tuple(t[:, None] if t.dim() == 1 else t for t in p)


@torch.no_grad()
def matgptq(W, H, count, *, l2=1.0, l4=1.0, kind="centre", feedback="dual", group=128, had=False,
            sigma_reg=0.03, seed=91426, nofb=False):
    """W [out,in], H [in,in] unnormalised Gram. Returns (W2, W4, info) dequantised in the original basis."""
    dev = torch.device("cuda")
    W = W.to(dev, torch.float32).clone()
    Hm = H.to(dev, torch.float32).clone() / count
    Hm.diagonal().add_(sigma_reg * torch.diag(Hm).mean().item())
    r, n = W.shape
    if had:
        torch.manual_seed(seed)
        Hb = had_blk(128, dev)
        su = torch.randn(n, device=dev).sign(); su[su == 0] = 1
        sv = torch.randn(r, device=dev).sign(); sv[sv == 0] = 1
        W = rot_out(rot_in(W, su, Hb), sv, Hb)
        Hm = rot_in(rot_in(Hm, su, Hb).T.contiguous(), su, Hb).T.contiguous()   # R^T H R
    Hinv = torch.cholesky_inverse(torch.linalg.cholesky(Hm.double())).float()
    U = torch.linalg.cholesky(Hinv.double(), upper=True).float()
    del Hinv
    W2 = W.clone(); W4 = W.clone() if feedback == "dual" else W2
    Q2 = torch.zeros_like(W); Q4 = torch.zeros_like(W)
    codes = torch.zeros((r, n), dtype=torch.uint8, device=dev)
    lw = l2 + l4
    for c1 in range(0, n, group):
        c2 = c1 + group
        b2 = W2[:, c1:c2].clone(); b4 = W4[:, c1:c2].clone() if feedback == "dual" else b2
        p = find_params(kind, b2, b4, l2, l4)
        v2, v4 = Grid(kind).values(_col(p))
        e2s = torch.zeros_like(b2); e4s = torch.zeros_like(b2)
        Ub = U[c1:c2, c1:c2]
        for i in range(group):
            x2 = b2[:, i]; x4 = b4[:, i]
            cost = l2 * (x2[:, None] - v2).square() + l4 * (x4[:, None] - v4).square()
            c = cost.argmin(1)
            q2 = v2.gather(1, c[:, None])[:, 0]; q4 = v4.gather(1, c[:, None])[:, 0]
            codes[:, c1 + i] = c.to(torch.uint8)
            Q2[:, c1 + i] = q2; Q4[:, c1 + i] = q4
            d = Ub[i, i]
            if nofb:
                continue
            e2 = (x2 - q2) / d; e4 = (x4 - q4) / d
            if feedback == "dual":
                b2[:, i:].addr_(e2, Ub[i, i:], alpha=-1); b4[:, i:].addr_(e4, Ub[i, i:], alpha=-1)
                e2s[:, i] = e2; e4s[:, i] = e4
            else:
                e = (l2 * e2 + l4 * e4) / lw
                b2[:, i:].addr_(e, Ub[i, i:], alpha=-1)
                e2s[:, i] = e
        W2[:, c2:].addmm_(e2s, U[c1:c2, c2:], alpha=-1)
        if feedback == "dual":
            W4[:, c2:].addmm_(e4s, U[c1:c2, c2:], alpha=-1)
    # fp16 storage of params: emulate by rounding the dequantised values' params? values are fp32 affine in
    # fp16 params; the error from fp16 params is negligible vs 2/4-bit error, so ignore.
    if had:
        Q2 = unrot_in(unrot_out(Q2, sv, Hb), su, Hb)
        Q4 = unrot_in(unrot_out(Q4, sv, Hb), su, Hb)
    g = Grid(kind)
    extra = (r + n) / (r * n) if had else 0.0          # sign bits
    info = dict(bpw2=2 + g.bits(2) / group + extra, bpw4=4 + g.bits(4) / group + extra)
    return Q2, Q4, info


def fit_expert(data, **kw):
    q2, q4, infos = [], [], []
    for i in range(3):
        H = data.H(i, normalized=False)
        a, b, info = matgptq(data.teacher[i], H, data.count, **kw)
        del H; torch.cuda.empty_cache()
        q2.append(a); q4.append(b); infos.append(info)
    return q2, q4, infos[0]
