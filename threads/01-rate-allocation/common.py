"""Shared helpers for thread 01 (rate allocation over block-LDL innovations)."""
import math, os
import torch
import numpy as np

ROOT = '/home/coder/git/orbit-duet'
GLM_SRC = '/tmp/orbit-duet-glm53-fp8'
MIMO_SRC = '/tmp/mimo-a100/data/jarrel/mimo-exl3-sequential/source'


def setup_gpu():
    torch.cuda.set_per_process_memory_fraction(12 / 80)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def stats_path(model, L, E):
    if model == 'mimo':
        return f'{ROOT}/runs/full55_statistics/l{L}_e{E}'
    return f'{ROOT}/runs/glm53_pilot_matched_l{L}/statistics/l{L}_e{E}'


def load_grams(model, L, E):
    d = torch.load(stats_path(model, L, E) + '.pt', weights_only=True, mmap=True)
    return d['grams'][0].clone(), d['grams'][1].clone(), [o.clone() for o in d['outputs']], d['metadata']


def hadamard(n, device, dtype):
    H = torch.ones(1, 1, dtype=dtype, device=device)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(n)


def rotate_H(H, seed=0, had=128, damp=0.025):
    """EXL3-style: damp by sigma_reg*mean(diag), random signs, blockwise 128-Hadamard (both sides)."""
    H = H.clone()
    H.diagonal().add_(damp * H.diagonal().mean())
    k = H.shape[0]
    g = torch.Generator(device='cpu').manual_seed(seed)
    su = (torch.randint(0, 2, (k,), generator=g) * 2 - 1).to(H.dtype).to(H.device)
    P = hadamard(had, H.device, H.dtype)
    H = H * su[None, :] * su[:, None]
    H = (H.view(k, k // had, had) @ P).view(k, k)
    H = (P @ H.view(k // had, had, k)).view(k, k)  # P symmetric
    return H, su


def rotate_W_in(W, su, had=128):
    """W: (k_in, n_out) EXL3 layout. Apply same input transform (P S W), so that tr(E^T H_rot E) is the proxy."""
    k, n = W.shape
    P = hadamard(had, W.device, W.dtype)
    W = W * su[:, None]
    return (P @ W.view(k // had, had, n)).view(k, n)


def unrotate_W_in(W, su, had=128):
    k, n = W.shape
    P = hadamard(had, W.device, W.dtype)
    W = (P @ W.view(k // had, had, n)).view(k, n)
    return W * su[:, None]


def block_ldl(H, b=16):
    """H = Lt D Lt^T with unit block-lower Lt. Returns Lt (with identity diag blocks) and D blocks (m,b,b)."""
    n = H.shape[0]; m = n // b
    L = torch.linalg.cholesky(H)
    DL = torch.diagonal(L.reshape(m, b, m, b), dim1=0, dim2=2).permute(2, 0, 1)  # (m,b,b)
    D = DL @ DL.transpose(1, 2)
    DLi = torch.linalg.inv(DL)
    Lt = torch.empty_like(L)
    Lv = L.view(n, m, b)
    Ltv = Lt.view(n, m, b)
    for i in range(m):
        Ltv[:, i, :] = Lv[:, i, :] @ DLi[i]
    return Lt, D


def waterfill(a, Rbar, lo=0.0, hi=8.0, iters=200):
    """min sum a_i 2^{-2R_i}  s.t. mean R_i = Rbar, lo<=R_i<=hi. Returns R (numpy)."""
    a = np.asarray(a, dtype=np.float64)
    la = 0.5 * np.log2(a)
    # R_i = clip(la_i - t)
    tl, th = la.min() - hi - 1, la.max() - lo + 1
    for _ in range(iters):
        t = 0.5 * (tl + th)
        R = np.clip(la - t, lo, hi)
        if R.mean() > Rbar: tl = t
        else: th = t
    return np.clip(la - 0.5 * (tl + th), lo, hi)


def greedy_int(a, Rbar, lo=1, hi=8, step=1.0, gain=None):
    """Integer (multiple of step) allocation by greedy marginal return under model a*2^{-2R}.
    total budget = Rbar*m exactly (Rbar multiple of step)."""
    a = np.asarray(a, dtype=np.float64); m = len(a)
    R = np.full(m, float(lo))
    budget = int(round((Rbar - lo) * m / step))
    import heapq
    f = (lambda ai, r: ai * 2.0 ** (-2 * r)) if gain is None else gain
    h = [(-(f(a[i], R[i]) - f(a[i], R[i] + step)), i) for i in range(m)]
    heapq.heapify(h)
    for _ in range(budget):
        _, i = heapq.heappop(h)
        R[i] += step
        if R[i] + step <= hi:
            heapq.heappush(h, (-(f(a[i], R[i]) - f(a[i], R[i] + step)), i))
    return R


def dist_model(a, R):
    a = np.asarray(a, np.float64)
    return float((a * 2.0 ** (-2 * np.asarray(R))).sum())


def amgm_db(a):
    a = np.asarray(a, np.float64)
    return 10 * math.log10(a.mean() / math.exp(np.log(a).mean()))
