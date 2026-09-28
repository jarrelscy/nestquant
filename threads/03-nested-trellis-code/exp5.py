# (a) short tail-biting trellis (32/64 weights) vs 256 tile; (b) J-style combined-window nested decode (thread 04)
import torch, math, json
torch.cuda.set_per_process_memory_fraction(12/80)
from trellis import *
from exllamav3.modules.quant.exl3_lib.quantize import quantize_tiles
torch.manual_seed(0)
db = lambda m, R: 10 * math.log10(m / 2 ** (-2 * R))
res = {}
def log(n, m, R): res[n] = (m, db(m, R)); print(f"{n:60s} mse={m:.6f} {db(m,R):+.3f} dB", flush=True)
x = torch.randn(512, 256, device='cuda')
C = mul1_codebook(16)[:, None]
for K, s in [(2, 1.0), (4, 0.925)]:
    for T in [256, 64, 32]:
        xb = (x * s).reshape(-1, T, 1)
        qs = []
        for i in range(0, xb.shape[0], 256 * 256 // T):
            q, _ = viterbi(xb[i:i + 256 * 256 // T], C, 16, K); qs.append(q)
        q = torch.cat(qs).reshape(x.shape) / s
        log(f"(a) native mul1 K{K} tail-biting length {T}", ((x - q) ** 2).mean().item(), K)
# (b) J4 / J3 with frozen native base
q2, idx = quantize_tiles(x.contiguous(), {"K": 2, "mul1": True})
b16 = idx.long() & 0xffff
log("base native K2 (256 tile)", ((x - q2) ** 2).mean().item(), 2)
M = mul1_codebook(16)
w = torch.arange(256, device='cuda')
# J4 ref window (my order): 4 steps x (p3,p4), newest in low bits: bits (2j+1)=p3_j, (2j)=p4_j for step j (0=newest)
p3 = sum((((w >> (2 * j + 1)) & 1) << j) for j in range(4)); p4 = sum((((w >> (2 * j)) & 1) << j) for j in range(4))
permJ4 = (p3 << 4) | p4
b8 = torch.arange(256, device='cuda')
CJ4 = M[(b8[:, None] << 8) | permJ4[None, :]].reshape(-1, 1)          # index = base8<<8 | w
ctx = (b16 & 255) << 8
best = None
for s in [0.9, 1.0, 1.1]:
    q, _ = viterbi((x * s).view(512, 256, 1), CJ4, 8, 2, ctx=ctx)
    m = ((x - q.view(512, 256) / s) ** 2).mean().item(); best = m if best is None else min(best, m)
log("(b) J4 frozen native base: 4b = mul1(8b base|4b P3|4b P4)", best, 4)
# J3: window 8b base | 8b P3, P3 1 bit/step
CJ3 = M[(b8[:, None] << 8) | w[None, :]].reshape(-1, 1)
best = None
for s in [0.9, 1.0, 1.1]:
    q, _ = viterbi((x * s).view(512, 256, 1), CJ3, 8, 1, ctx=ctx)
    m = ((x - q.view(512, 256) / s) ** 2).mean().item(); best = m if best is None else min(best, m)
log("(b) J3 frozen native base: 3b = mul1(8b base|8b P3)", best, 3)
json.dump(res, open('exp5.json', 'w'), indent=1)
