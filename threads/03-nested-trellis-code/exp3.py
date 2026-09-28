# Refinement-stage variants with own Viterbi on the native EXL3 K=2 base residual.
import torch, json, math, sys, time
torch.cuda.set_per_process_memory_fraction(12/80)
from trellis import *
from exllamav3.modules.quant.exl3_lib.quantize import quantize_tiles
torch.manual_seed(0)
NTR, NTE = 512, 512
x = torch.randn(NTR + NTE, 256, device='cuda')
q2, idx2 = quantize_tiles(x.contiguous(), {"K": 2, "mul1": True})
base_w = idx2.long() & 0xffff
e2 = x - q2
etr, ete = e2[:NTR], e2[NTR:]
def db(m, R): return 10 * math.log10(m / 2 ** (-2 * R))
res = {}
def log(name, m, R, extra=None):
    res[name] = dict(mse=m, db_rd=db(m, R), **(extra or {}))
    print(f"{name:45s} L{R}: mse={m:.6f} {db(m,R):+.3f} dB  {extra or ''}", flush=True)
sd = e2.std().item()

def scaled_fit(t, C, L, k, V=1, grid=None, bs=256, ctx=None):
    best = None
    for s in grid:
        q, w = fit(t * s, C, L, k, V, bs=bs, ctx=ctx)
        m = ((t - q / s) ** 2).mean().item()
        if best is None or m < best[0]: best = (m, s, q / s, w)
    return best

def lloyd_lut(ttr, tte, L, k, V, iters, init, bs=256, hashf=None, Q=None):
    """Learned codebook: C[w] (direct, 2^L entries) or LUT[hash(w)] (2^Q entries, with sign bit)."""
    C = init.clone()
    for it in range(iters + 1):
        Cfull = C if hashf is None else hashf(C)
        q, w = fit(tte, Cfull, L, k, V, bs=bs)
        mte = ((tte - q) ** 2).mean().item()
        if it == iters: break
        q, w = fit(ttr, Cfull, L, k, V, bs=bs)
        tv = ttr.view(-1, 256 // V, V).reshape(-1, V)
        wv = w.reshape(-1)
        if hashf is None:
            num = torch.zeros(2 ** L, V, device='cuda').index_add_(0, wv, tv)
            cnt = torch.zeros(2 ** L, device='cuda').index_add_(0, wv, torch.ones_like(wv, dtype=torch.float))
        else:
            ix, sg = hashf.idx(wv)
            num = torch.zeros(2 ** Q, V, device='cuda').index_add_(0, ix, tv * sg[:, None])
            cnt = torch.zeros(2 ** Q, device='cuda').index_add_(0, ix, torch.ones_like(sg))
        upd = cnt > 0
        C[upd] = num[upd] / cnt[upd, None]
        print(f"   lloyd it{it} test mse {mte:.6f}", flush=True)
    return mte, C

which = sys.argv[1]
t0 = time.time()
if which == 'mul1L':
    for L in [12, 16, 20]:
        for K, R in [(2, 4), (1, 3)]:
            C = mul1_codebook(L)[:, None]
            bs = 256 if L <= 16 else 16
            tt = time.time()
            m, s, _, _ = scaled_fit(ete, C, L, K, grid=[(0.9 + 0.05 * i) / sd for i in range(5)], bs=bs)
            log(f"res mul1 K{K} L{L}", m, R, dict(scale=s * sd, fit_s_per_1Mw=(time.time() - tt) / 5 / (NTE * 256 / 1e6)))
elif which == 'lut':
    for K, R in [(2, 4), (1, 3)]:
        for L in [8, 10, 12]:
            init = (mul1_codebook(L) * sd)[:, None]
            m, C = lloyd_lut(etr, ete, L, K, 1, 8, init)
            log(f"res learned-LUT direct K{K} L{L} ({2**L} halfs)", m, R)
elif which == 'ctx':
    # refinement codebook index = ref window XOR hash(base window at same position)
    for K, R in [(2, 4), (1, 3)]:
        C = mul1_codebook(16)[:, None]
        ctx = ((base_w[NTR:] * 0x9E3779B1) >> 16) & 0xffff
        m, s, _, _ = scaled_fit(ete, C, 16, K, grid=[(0.9 + 0.05 * i) / sd for i in range(5)], ctx=ctx)
        log(f"res mul1 K{K} L16 ctx=base-window hash", m, R)
elif which == 'hyb':
    # QTIP-HYB-like V=2: window L=16, k=2V=4 bits/step; value = sign * LUT[idx], idx/sign from hash of window
    class Hyb:
        def __init__(s, L, Q): s.L, s.Q = L, Q
        def idx(s, w):
            h = (w * w + w) & 0xFFFFFFFF
            h = (h * 0x83DCD12D) & 0xFFFFFFFF
            ix = (h >> (32 - s.Q)) & (2 ** s.Q - 1)
            sg = 1.0 - 2.0 * ((h >> (31 - s.Q)) & 1).float()
            return ix, sg
        def __call__(s, lut):
            w = torch.arange(2 ** s.L, device='cuda')
            ix, sg = s.idx(w)
            return lut[ix] * sg[:, None]
    for tgt_name, (ttr, tte, R) in {'base': (x[:NTR], x[NTR:], 2), 'res': (etr, ete, 4)}.items():
        for Q in [8, 9, 10]:
            hf = Hyb(16, Q)
            g = torch.Generator(device='cuda').manual_seed(1)
            init = torch.randn(2 ** Q, 2, device='cuda', generator=g) * (ttr.std().item())
            m, C = lloyd_lut(ttr, tte, 16, 4, 2, 8, init, hashf=hf, Q=Q)
            log(f"HYB V2 {tgt_name} L16 Q{Q} ({2**Q}x half2)", m, R)
            if tgt_name == 'base': torch.save(C, f'/tmp/nestquant/03-nested-trellis-code/hyb_base_Q{Q}.pt')
    # V=2 direct LUT refinement L=12 (4096 half2 = 16KB)
    for L in [8, 12]:
        init = torch.randn(2 ** L, 2, device='cuda') * sd
        m, C = lloyd_lut(etr, ete, L, 4, 2, 8, init)
        log(f"res V2 direct-LUT L{L} ({2**L}x half2)", m, 4)
json.dump(res, open(f'exp3_{which}.json', 'w'), indent=1)
print('total s', time.time() - t0)
