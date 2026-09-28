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
IX = (h >> (15 - Q)) & (2 ** Q - 1)
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
json.dump(res, open('exp4b.json', 'w'), indent=1)
