# Generic tail-biting bitshift-trellis Viterbi (torch, GPU).
# Step t consumes k bits, emits V values from codebook C[window] (window = last L bits).
import torch, math
M_MUL1 = 0x83DCD12D
M_MCG = 0xCBAC1FED

def mul1_codebook(L, device='cuda', mult=M_MUL1):
    idx = torch.arange(2**L, device=device, dtype=torch.int64)
    p = (idx * mult) & 0xFFFFFFFF
    s = (p & 255) + ((p >> 8) & 255) + ((p >> 16) & 255) + ((p >> 24) & 255)
    v = (s.float() + 1024.0).half().float() * float(torch.tensor(0x1eee, dtype=torch.int16).view(torch.half)) \
        + float(torch.tensor(0xc931 - 65536, dtype=torch.int16).view(torch.half))
    return v.half().float()

def viterbi(x, C, L, k, s0=None, wfun=None, ctx=None):
    """x: [B, T, V] targets. C: [2^L, V] codebook (or callable per step). Returns (q [B,T,V], windows [B,T]).
    ctx: optional [B, T] int64 XOR-ed into window before codebook lookup (per-position context).
    Two-pass tail-biting: pass 1 free, pass 2 forced start/end state from pass-1 end."""
    B, T, V = x.shape
    S = 2**L; Sp = 2**(L - k); nk = 2**k
    dev = x.device
    def run(s0):
        cost = torch.zeros(B, S, device=dev)
        if s0 is not None:
            # prev window w has low (L-k) bits == s0
            mask = torch.full((B, nk, Sp), float('inf'), device=dev)
            mask.scatter_(2, s0.view(B, 1, 1).expand(B, nk, 1), 0.0)
            cost = mask.view(B, S)
        bp = torch.empty(T, B, Sp, dtype=torch.uint8, device=dev)
        for t in range(T):
            m, a = cost.view(B, nk, Sp).min(dim=1)       # over dropped top bits
            bp[t] = a.to(torch.uint8)
            xt = x[:, t, :]                                 # [B,V]
            if ctx is None:
                d = ((xt[:, None, :] - C[None, :, :]) ** 2).sum(-1)   # [B,S]
            else:
                w = torch.arange(S, device=dev)[None, :] ^ ctx[:, t:t+1]
                d = ((xt[:, None, :] - C[w]) ** 2).sum(-1)
            cost = (m[:, :, None] + d.view(B, Sp, nk)).view(B, S)
        return cost, bp
    def backtrack(cost, bp, s0):
        if s0 is not None:
            # final window low (L-k) bits must equal s0
            cand = torch.arange(nk, device=dev)[None, :] * 0  # placeholder
            full = cost.view(B, Sp, nk)  # index w = hi*nk + lo? no: w = (w>>k)<<k | new ; low L-k bits of w != w>>k
            wv = torch.arange(S, device=dev)
            ok = (wv[None, :] & (Sp - 1)) == s0[:, None]
            cost = torch.where(ok, cost, torch.full_like(cost, float('inf')))
        w = cost.argmin(dim=1)
        ws = torch.empty(B, T, dtype=torch.int64, device=dev)
        for t in range(T - 1, -1, -1):
            ws[:, t] = w
            top = bp[t].gather(1, (w >> k).view(B, 1)).view(B).long()
            w = (top << (L - k)) | (w >> k)
        return ws
    W = min(64, T)
    xs = x
    x = torch.cat([xs[:, T - W:], xs, xs[:, :W]], dim=1)
    ctx_s = ctx
    if ctx is not None: ctx = torch.cat([ctx_s[:, T - W:], ctx_s, ctx_s[:, :W]], dim=1)
    T2 = T; T = T2 + 2 * W
    cost, bp = run(None)
    ws = backtrack(cost, bp, None)
    s0 = ws[:, W - 1] & (Sp - 1)
    del bp
    x = xs; ctx = ctx_s; T = T2
    cost, bp = run(s0)
    ws = backtrack(cost, bp, s0)
    del bp
    widx = ws if ctx is None else ws ^ ctx
    q = C[widx]
    return q, ws

def fit(x, C, L, k, V=1, bs=256, ctx=None):
    """x: [N, 256] tiles. returns q [N,256], windows [N, 256/V]"""
    N = x.shape[0]; T = 256 // V
    qs, ws = [], []
    for i in range(0, N, bs):
        xb = x[i:i+bs].view(-1, T, V)
        cb = None if ctx is None else ctx[i:i+bs]
        q, w = viterbi(xb, C, L, k, ctx=cb)
        qs.append(q.reshape(-1, 256)); ws.append(w)
    return torch.cat(qs), torch.cat(ws)
