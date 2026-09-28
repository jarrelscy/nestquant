"""Pattern-rate bitshift trellis (EXL3 frac generalisation): step i shifts in D(i) = KA + bit(i mod 16) of MASK bits,
state_i = ((state_{i-1} << D(i)) | b_i) & 0xFFFF, value = mul1_lut[state_i], 256-step tail-biting ring
(two-pass trick as EXL3). K = KA + popcount(MASK)/16; every tile is 16K uint16 words, so any n/16 rate
packs into whole words. Decoder: same mul1 op count as integer K (shift amounts are compile-time
constants per unrolled position)."""
import torch
import harness as h

_LUT = None


def lut():
    global _LUT
    if _LUT is None:
        _LUT = h.codebook_lut('mul1').float().cuda()
    return _LUT


def pattern(K):
    """(KA, MASK) with popcount spread evenly over the 16-step period."""
    ka = int(K // 1)
    n = round((K - ka) * 16)
    assert abs(ka + n / 16 - K) < 1e-9, K
    mask = 0
    for j in range(n):
        mask |= 1 << int((j + 0.5) * 16 / n)
    return ka, mask


def steps(K, L=256):
    ka, mask = pattern(K)
    return [ka + ((mask >> (i & 15)) & 1) for i in range(L)]


@torch.no_grad()
def _run(w, D):
    T, L = w.shape
    dev = w.device
    V = lut()
    inf = float('inf')
    backs = [None] * L

    def forward(roll, start):
        if start is None:
            cost = torch.zeros(T, 65536, device=dev)
        else:
            cost = torch.full((T, 65536), inf, device=dev)
            cost.scatter_(1, start[:, None], 0.)
        for i in range(L):
            ri = (i + roll) % L
            k = D[ri]
            E = 1 << (16 - k)
            mn, top = cost.view(T, 1 << k, E).min(1)                     # prev = (top << (16-k)) | e
            backs[ri] = top.to(torch.uint8)
            d = (V[None] - w[:, ri, None]).square()                       # [T, 65536]
            cost = (d.view(T, E, 1 << k) + mn[:, :, None]).view(T, 65536)  # s = (e << k) | b
        return cost

    def trace(roll, s, stop_at_zero):
        out = torch.empty((T, L), dtype=torch.int64, device=dev)
        tt = torch.arange(T, device=dev)
        for i in range(L - 1, -1, -1):
            ri = (i + roll) % L
            k = D[ri]
            out[:, ri] = s
            e = s >> k
            top = backs[ri][tt, e].long()
            s = (top << (16 - k)) | e
            if stop_at_zero and ri == 0:
                break
        return out, s                                                    # s = state before position 0

    c = forward(L // 2, None)
    _, start = trace(L // 2, c.argmin(1), True)
    c = forward(0, start)
    idx, _ = trace(0, start, False)                                     # ring closes on start
    return V[idx], idx


def quantize_tiles(tiles, K, chunk=192):
    D = steps(K, tiles.shape[1])
    qs, ids = [], []
    for a in range(0, tiles.shape[0], chunk):
        q, i = _run(tiles[a:a + chunk].float(), D)
        qs.append(q); ids.append(i)
    return torch.cat(qs), torch.cat(ids)


if __name__ == '__main__':
    h.gpu_cap(12)
    torch.manual_seed(0)
    x = torch.randn(512, 256, device='cuda')
    for K in (2, 2.5, 1.5):
        qc, _ = h.ExtTileQuantizer('mul1')(x.clone(), K)
        qt, _ = quantize_tiles(x, K)
        print(K, 'cuda', float((qc.float() - x).square().mean()), 'torch', float((qt - x).square().mean()), 'agree', float((qc.float() == qt).float().mean()))
    for K in (1.75, 1.875, 2.125, 2.25, 2.375):
        qt, _ = quantize_tiles(x, K)
        print(K, pattern(K), 'torch', float((qt - x).square().mean()))
