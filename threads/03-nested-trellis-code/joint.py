# Joint (product-state) Viterbi for a 2-level nested trellis:
#   level-2 value  f2(b16)            b16 = last 8 base symbols (2 bits each)  -> identical to native EXL3 K=2 decode
#   level-4 value  f2(b16) + d*g(W)   W = (b16<<8)|r8, r8 = last 4 refinement symbols (2 bits each)
# cost per weight: alpha*(x-f2)^2 + beta*(x-f2-d*g)^2 ; joint state = (b16 low14, r8 low6) = 2^20 states, 16 branches.
import torch, math, sys, json, time
torch.cuda.set_per_process_memory_fraction(12/80)
from trellis import mul1_codebook, viterbi
from exllamav3.modules.quant.exl3_lib.quantize import quantize_tiles
dev = 'cuda'
F2 = mul1_codebook(16)                 # [65536]

def joint_viterbi(x, G, alpha, beta, s0=None):
    """x [B,T]; G [2^24] level-4 increment table (already scaled). returns b16 [B,T], r8 [B,T]"""
    B, T = x.shape
    NS = 2 ** 24
    f2 = F2.view(1, 2 ** 16, 1).expand(1, 2 ** 16, 256).reshape(1, NS)     # f2 per W
    g = G.view(1, NS)
    def run(xx, s0):
        TT = xx.shape[1]
        cost = torch.zeros(B, NS, device=dev)
        if s0 is not None:
            cost = torch.full((B, NS), float('inf'), device=dev)
            # previous W with (b16 low14, r8 low6) == s0: W = tb<<22 | lb<<8 | tr<<6 | lr
            lb = s0 >> 6; lr = s0 & 63
            for tb in range(4):
                for tr in range(4):
                    cost[torch.arange(B), (tb << 22) | (lb << 8) | (tr << 6) | lr] = 0.0
        bp = torch.empty(TT, B, 2 ** 20, dtype=torch.uint8, device=dev)
        for t in range(TT):
            c = cost.view(B, 4, 2 ** 14, 4, 64)
            m3, a3 = c.min(dim=3)                 # [B,4,2^14,64]
            m, a1 = m3.min(dim=1)                 # [B,2^14,64]
            a3s = a3.gather(1, a1.unsqueeze(1)).squeeze(1)
            bp[t] = (a1 * 4 + a3s).view(B, -1).to(torch.uint8)
            xt = xx[:, t:t + 1]
            e = xt - f2
            d = alpha * e * e + beta * (e - g) ** 2           # [B, NS], index W' = lb<<10 | b<<8 | lr<<2 | r
            cost = (m.view(B, 2 ** 14, 1, 64, 1) + d.view(B, 2 ** 14, 4, 64, 4)).view(B, NS)
            del d, e, c, m3, a3
        return cost, bp
    def backtrack(cost, bp, s0):
        TT = bp.shape[0]
        if s0 is not None:
            Wv = torch.arange(NS, device=dev)
            st = ((Wv >> 8) & (2 ** 14 - 1)) << 6 | (Wv & 63)
            cost = torch.where(st[None] == s0[:, None], cost, torch.full_like(cost, float('inf')))
        W = cost.argmin(1)
        Ws = torch.empty(B, TT, dtype=torch.int64, device=dev)
        for t in range(TT - 1, -1, -1):
            Ws[:, t] = W
            b16 = W >> 8; r8 = W & 255
            st = ((b16 >> 2) << 6) | (r8 >> 2)
            a = bp[t].gather(1, st.view(B, 1)).view(B).long()
            tb, tr = a >> 2, a & 3
            W = (tb << 22) | ((b16 >> 2) << 8) | (tr << 6) | (r8 >> 2)
        return Ws
    Wu = 32
    xw = torch.cat([x[:, T - Wu:], x, x[:, :Wu]], 1)
    cost, bp = run(xw, None); Ws = backtrack(cost, bp, None); del bp, cost
    W = Ws[:, Wu - 1]
    s0 = (((W >> 8) & (2 ** 14 - 1)) << 6) | (W & 63)
    cost, bp = run(x, s0); Ws = backtrack(cost, bp, s0); del bp, cost
    return Ws

def evalW(x, Ws, G):
    q2 = F2[Ws >> 8]; q4 = q2 + G[Ws]
    return ((x - q2) ** 2).mean().item(), ((x - q4) ** 2).mean().item()

if __name__ == '__main__':
    torch.manual_seed(0)
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    x = torch.randn(N, 256, device=dev)
    db = lambda m, R: 10 * math.log10(m / 2 ** (-2 * R))
    Wi = torch.arange(2 ** 24, device=dev, dtype=torch.int64)
    res = {}
    for gname in ['hash24', 'lut8']:
        if gname == 'hash24':
            G0 = mul1_codebook(24).float() if False else None
            p = ((Wi * 0x9E3779B1) & 0xFFFFFFFF)
            p = (p * 0x83DCD12D) & 0xFFFFFFFF
            s = (p & 255) + ((p >> 8) & 255) + ((p >> 16) & 255) + ((p >> 24) & 255)
            G0 = (s.float() - 510.0) / 147.8
        else:
            G0 = mul1_codebook(8)[Wi & 255]
        # sequential reference: native base (exl3) then refinement viterbi with base fixed
        q2, idx = quantize_tiles(x.contiguous(), {"K": 2, "mul1": True})
        b16 = idx.long() & 0xffff
        e2 = x - q2
        best = None
        for dl in [0.22, 0.25, 0.28]:
            ctx = b16 << 8
            q, w = viterbi(e2.view(N, 256, 1), (G0 * dl)[:, None], 8, 2, ctx=ctx)
            m4 = ((e2 - q.view(N, 256)) ** 2).mean().item()
            if best is None or m4 < best[0]: best = (m4, dl)
        D2 = (e2 ** 2).mean().item()
        print(f"[{gname}] sequential: D2={D2:.6f} ({db(D2,2):+.3f}) D4={best[0]:.6f} ({db(best[0],4):+.3f}) delta={best[1]}", flush=True)
        res[f"{gname}_seq"] = (D2, best[0])
        dl = best[1]
        for (a, b) in [(1, 1), (1, 4), (1, 16), (0, 1)]:
            t0 = time.time()
            Ws = torch.cat([joint_viterbi(x[i:i + 4], G0 * dl, a, b) for i in range(0, N, 4)])
            D2, D4 = evalW(x, Ws, G0 * dl)
            print(f"[{gname}] joint a={a} b={b}: D2={D2:.6f} ({db(D2,2):+.3f}) D4={D4:.6f} ({db(D4,4):+.3f})  {time.time()-t0:.0f}s", flush=True)
            res[f"{gname}_joint_{a}_{b}"] = (D2, D4)
    json.dump(res, open(f'joint_N{N}.json', 'w'), indent=1)
