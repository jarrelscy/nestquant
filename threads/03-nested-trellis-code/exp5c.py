import torch, math, json
torch.cuda.set_per_process_memory_fraction(12/80)
from trellis import *
torch.manual_seed(0)
x = torch.randn(512, 256, device='cuda'); C = mul1_codebook(16)[:, None]
out = {}
for K, s in [(2, 1.0), (4, 0.925)]:
    for T in [128]:
        xb = (x * s).reshape(-1, T, 1); qs = []
        for i in range(0, xb.shape[0], 512):
            q, _ = viterbi(xb[i:i + 512], C, 16, K); qs.append(q)
        q = torch.cat(qs).reshape(x.shape) / s; m = ((x - q) ** 2).mean().item()
        out[f"K{K}_T{T}"] = m; print(K, T, m, 10 * math.log10(m / 2 ** (-2 * K)))
json.dump(out, open('exp5c.json', 'w'))
