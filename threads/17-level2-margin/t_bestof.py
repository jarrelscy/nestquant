"""i.i.d. Gaussian: per-tile best-of-N among affine codebook variants (sign flip, gain) with the CUDA mul1 Viterbi."""
from core17 import *
torch.manual_seed(0)
x = torch.randn(8192, 256, device="cuda")
g0 = 0.97   # approx best global gain
def mse_at(s, b=0.0):
    q, _ = EXT((x - b) * s, 2); return ((q / s + b - x) ** 2).mean(1)
best = None
for g in [0.9, 0.95, 1.0, 1.05]:
    m = mse_at(g).mean().item(); print("gain", g, m)
base = mse_at(1.0)
print("base", base.mean().item())
variants = {"sign": [(1.0, 1), (1.0, -1)],
            "gain3": [(0.92, 1), (1.0, 1), (1.08, 1)],
            "sign x gain2": [(0.95, 1), (1.05, 1), (0.95, -1), (1.05, -1)],
            "sign x gain4": [(g, s) for g in [0.9, 0.97, 1.03, 1.1] for s in [1, -1]],
            "sign x gain8": [(g, s) for g in [0.86, 0.9, 0.94, 0.98, 1.02, 1.06, 1.1, 1.14] for s in [1, -1]]}
cache = {}
def m(g, s):
    if (g, s) not in cache: cache[(g, s)] = mse_at(g * s)
    return cache[(g, s)]
for name, vs in variants.items():
    M = torch.stack([m(g, s) for g, s in vs]).min(0).values.mean().item()
    bits = math.log2(len(vs))
    print(f"{name:14s} N={len(vs):2d} bits/tile {bits:.0f} mse {M:.5f}  gain {10*math.log10(base.mean().item()/M):.3f} dB  "
          f"break-even {6.02*bits/256:.3f} dB")
# offsets
for bset in [[-0.05, 0.05], [-0.1, 0, 0.1]]:
    M = torch.stack([mse_at(1.0, b) for b in bset]).min(0).values.mean().item(); print("offsets", bset, M, 10*math.log10(base.mean().item()/M))
