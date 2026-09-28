"""Pattern-rate residual trellis (thread 14's patvit/t12pat, vendored so the production encoder does not depend on T14's
working dir). Step i (Viterbi order) shifts in D(i) bits, state_i = ((state_{i-1} << D(i)) | b_i) & 0xFFFF,
value = mul1 LUT, 256-step tail-biting ring (EXL3 two-pass trick). Kernel convention: w_p = KA + ((MASK >> (p%16)) & 1)
at ring position p = (-i) mod 256, so Viterbi step i uses w_{(-i) mod 256}. Ring bits = 16 * K (whole u16 words).
"""
import torch
import harness as h
import nq_decode as D

# T14-verified patterns (kernel masks) + LDLQ drift factors (thread 14 fits)
NEW_PATTERNS = {1.875: (1, 0xFEFE), 1.9375: (1, 0xFFFE), 2.3125: (2, 0x9248)}      # 2.25 = (2, 0x8888) already present
DRIFT = {1.875: 1.02225, 1.9375: 1.0201, 2.25: 1.0135, 2.3125: 1.0124}
_LUT = None


def install():
    for K, v in NEW_PATTERNS.items():
        D.PATTERNS.setdefault(K, v)
    Qm = h._ex()
    for K, d in DRIFT.items():
        Qm.LDLQ_DRIFT.setdefault(K, d)


def is_pat(K):
    """True if exllamav3's CUDA Viterbi cannot do K (only integers and (KA, 0xAAAA) half rates)."""
    K = float(K)
    return not (K.is_integer() or (2 * K).is_integer())


def vsteps(K):
    KA, MASK = D.PATTERNS[float(K)]
    return [KA + ((MASK >> ((-i) % 16)) & 1) for i in range(256)]


@torch.no_grad()
def _run(w, Dst):
    global _LUT
    if _LUT is None:
        _LUT = h.codebook_lut("mul1").float().cuda()
    V = _LUT
    T, L = w.shape
    dev = w.device
    inf = float("inf")
    backs = [None] * L

    def forward(roll, start):
        if start is None:
            cost = torch.zeros(T, 65536, device=dev)
        else:
            cost = torch.full((T, 65536), inf, device=dev)
            cost.scatter_(1, start[:, None], 0.)
        for i in range(L):
            ri = (i + roll) % L
            k = Dst[ri]
            E = 1 << (16 - k)
            mn, top = cost.view(T, 1 << k, E).min(1)
            backs[ri] = top.to(torch.uint8)
            d = (V[None] - w[:, ri, None]).square()
            cost = (d.view(T, E, 1 << k) + mn[:, :, None]).view(T, 65536)
        return cost

    def trace(roll, s, stop_at_zero):
        out = torch.empty((T, L), dtype=torch.int64, device=dev)
        tt = torch.arange(T, device=dev)
        for i in range(L - 1, -1, -1):
            ri = (i + roll) % L
            k = Dst[ri]
            out[:, ri] = s
            e = s >> k
            top = backs[ri][tt, e].long()
            s = (top << (16 - k)) | e
            if stop_at_zero and ri == 0:
                break
        return out, s

    c = forward(L // 2, None)
    _, start = trace(L // 2, c.argmin(1), True)
    forward(0, start)
    idx, _ = trace(0, start, False)
    return V[idx], idx


def patq(tiles, K, chunk=128):
    """quantizer(tiles [R,256], K) -> (values, states) in Viterbi order (ExtTileQuantizer signature)."""
    St = vsteps(K)
    q, idx = [], []
    for a in range(0, tiles.shape[0], chunk):
        v, i = _run(tiles[a:a + chunk].float(), St)
        q.append(v); idx.append(i)
    return torch.cat(q), torch.cat(idx)
