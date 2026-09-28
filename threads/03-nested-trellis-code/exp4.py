# Full chains: HYB(V2) base + HYB(V2) ref with the exact decode used for SASS counting; K1+K1 with context hash.
import torch, json, math, sys
torch.cuda.set_per_process_memory_fraction(12/80)
from trellis import *
from exllamav3.modules.quant.exl3_lib.quantize import quantize_tiles
torch.manual_seed(0)
NTR, NTE = 512, 512
x = torch.randn(NTR + NTE, 256, device='cuda')
db = lambda m, R: 10 * math.log10(m / 2 ** (-2 * R))
res = {}
def log(n, m, R):
    res[n] = (m, db(m, R)); print(f"{n:50s} L{R} mse={m:.6f} {db(m,R):+.3f} dB", flush=True)
Q = 9
w = torch.arange(2 ** 16, device='cuda')
h = (w * w + w) & 0xFFFFFFFF
IX = (h >> (31 - Q)) & (2 ** Q - 1)
SG = torch.stack([1.0 - 2.0 * ((h >> 15) & 1).float(), 1.0 - 2.0 * ((h >> 31) & 1).float()], 1)  # low half, high half
def full(lut): return lut[IX] * SG
def lloyd(t_tr, t_te, iters=10, seed=1):
    g = torch.Generator(device='cuda').manual_seed(seed)
    lut = torch.randn(2 ** Q, 2, device='cuda', generator=g) * t_tr.std().item()
    for it in range(iters):
        q, ws = fit(t_tr, full(lut), 16, 4, 2)
        tv = t_tr.view(-1, 2); wv = ws.reshape(-1)
        num = torch.zeros(2 ** Q, 2, device='cuda').index_add_(0, IX[wv], tv * SG[wv])
        cnt = torch.zeros(2 ** Q, device='cuda').index_add_(0, IX[wv], torch.ones_like(wv, dtype=torch.float))
        ok = cnt > 0; lut[ok] = num[ok] / cnt[ok, None]
    return lut
lb = lloyd(x[:NTR], None)
qb_tr, _ = fit(x[:NTR], full(lb), 16, 4, 2); qb_te, _ = fit(x[NTR:], full(lb), 16, 4, 2)
log("HYB base (exact decode)", ((x[NTR:] - qb_te) ** 2).mean().item(), 2)
e_tr, e_te = x[:NTR] - qb_tr, x[NTR:] - qb_te
lr = lloyd(e_tr, None, seed=2)
qr, _ = fit(e_te, full(lr), 16, 4, 2)
log("HYB base + HYB ref -> L4", ((e_te - qr) ** 2).mean().item(), 4)
# K1+K1 with context hash on the mul1 base
q2, idx = quantize_tiles(x[NTR:].contiguous(), {"K": 2, "mul1": True})
e2 = x[NTR:] - q2; sd = e2.std().item()
C = mul1_codebook(16)[:, None]
b16 = idx.long() & 0xffff
ctx = ((b16 * 0x9E3779B1) >> 16) & 0xffff
best = None
for s in [1.0 / sd, 1.05 / sd, 1.1 / sd, 1.15 / sd]:
    q, w3 = fit(e2 * s, C, 16, 1, ctx=ctx); m = ((e2 - q / s) ** 2).mean().item()
    if best is None or m < best[0]: best = (m, s, q / s, w3)
log("mul1 base + K1 ctx -> L3", best[0], 3)
e3 = e2 - best[2]; sd3 = e3.std().item()
ctx4 = ((best[3] * 0x85EBCA6B) >> 16) & 0xffff ^ ctx
best4 = None
for s in [1.0 / sd3, 1.05 / sd3, 1.1 / sd3, 1.15 / sd3]:
    q, _ = fit(e3 * s, C, 16, 1, ctx=ctx4); m = ((e3 - q / s) ** 2).mean().item()
    if best4 is None or m < best4[0]: best4 = (m, s)
log("mul1 base + K1 ctx + K1 ctx(P3,base) -> L4", best4[0], 4)
json.dump(res, open('exp4.json', 'w'), indent=1)
