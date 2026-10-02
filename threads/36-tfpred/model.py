"""nq-tfpred multi-window expert-use predictor (transformer over expert tokens; raw routing history, no hand features).

Per refresh (end of a 16-token block) and per (layer, expert) token:
  input  = log1p of raw hit (or salience) counts in the last F=32 blocks and the last C=32 chunks of 8 blocks, the request's
           prefill hit fraction, + learned (layer, expert) and layer embeddings
  mixer  = per block: self-attention over the 256 experts of each layer; attention across layers on per-layer summaries;
           cross-attention to a context (answer/think fraction of the last 32 blocks, request position, and with content=True
           the last 64 token ids and block-mean projections of the MoE input at 8 layers for the last 4 blocks); FFN
  output = log rate (hits per token) in each window WIN (data.WIN, relative to the refresh): W=6."""
import math, torch, torch.nn as nn, torch.nn.functional as Fn

NL, NE, F, C = 75, 256, 32, 32


class MHA(nn.Module):
    def __init__(s, d, nh):
        super().__init__(); s.nh = nh; s.q = nn.Linear(d, d); s.kv = nn.Linear(d, 2 * d); s.o = nn.Linear(d, d)
        nn.init.zeros_(s.o.weight); nn.init.zeros_(s.o.bias)
    def forward(s, x, c):
        B, T, d = x.shape; S = c.shape[1]; h = s.nh
        q = s.q(x).view(B, T, h, d // h).transpose(1, 2)
        k, v = s.kv(c).view(B, S, 2, h, d // h).permute(2, 0, 3, 1, 4)
        return s.o(Fn.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, T, d))


class Block(nn.Module):
    def __init__(s, d, nh, ctx):
        super().__init__()
        s.n1 = nn.LayerNorm(d); s.sa = MHA(d, nh)
        s.n2 = nn.LayerNorm(d); s.la = MHA(d, nh)                     # across-layer mixing on layer summaries
        s.ctx = ctx
        if ctx: s.n3 = nn.LayerNorm(d); s.ca = MHA(d, nh)
        s.n4 = nn.LayerNorm(d); s.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        nn.init.zeros_(s.ff[2].weight); nn.init.zeros_(s.ff[2].bias)
    def forward(s, x, c):                                             # x [B,NL,NE,d], c [B,S,d]
        B, L, E, d = x.shape
        h = s.n1(x).view(B * L, E, d); x = x + s.sa(h, h).view(B, L, E, d)
        m = s.n2(x.mean(2)); x = x + s.la(m, m)[:, :, None]
        if s.ctx:
            h = s.n3(x).view(B, L * E, d); x = x + s.ca(h, c).view(B, L, E, d)
        return x + s.ff(s.n4(x))


class TFPred(nn.Module):
    def __init__(s, d=64, nh=4, nblk=2, W=6, content=False, P=8, vocab=155008, scale=None):
        super().__init__()
        s.cfg = dict(d=d, nh=nh, nblk=nblk, W=W, content=content, P=P, vocab=vocab)
        s.content = content
        s.eemb = nn.Parameter(torch.zeros(NL, NE, d)); s.lemb = nn.Parameter(torch.randn(NL, 1, d) * 0.02)
        s.inp = nn.Sequential(nn.Linear(F + C + 1, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        s.register_buffer('scale', torch.ones(NL) if scale is None else torch.as_tensor(scale, dtype=torch.float32))
        s.g = nn.Linear(F + 1, d)                                     # global context token
        if content:
            s.temb = nn.Embedding(vocab, d); s.tpos = nn.Parameter(torch.randn(64, d) * 0.02)
            s.hpj = nn.Linear(256, d); s.hpos = nn.Parameter(torch.randn(4, P, d) * 0.02)
        s.blocks = nn.ModuleList([Block(d, nh, True) for _ in range(nblk)])
        s.nf = nn.LayerNorm(d); s.head = nn.Linear(d, W)
        nn.init.zeros_(s.head.weight); nn.init.constant_(s.head.bias, math.log(8 / 256))

    def forward(s, fine, coarse, pf, ans, rpos, tok=None, tokm=None, hp=None):
        """fine [B,F,NL,NE], coarse [B,C,NL,NE] (raw counts; salience pre-divided by scale), pf [B,NL,NE] per-token fraction,
        ans [B,F] answer fraction per block, rpos [B] blocks since request start. returns log rate [B,NL,NE,W]."""
        sc = s.scale[None, None, :, None]
        x = torch.cat([torch.log1p(fine / sc).permute(0, 2, 3, 1), torch.log1p(coarse / sc).permute(0, 2, 3, 1) * 0.5,
                       torch.log1p(pf * 16)[..., None]], -1)
        x = s.inp(x) + s.eemb[None] + s.lemb[None]
        ctx = [s.g(torch.cat([ans, torch.log1p(rpos)[:, None] / 5], -1))[:, None]]
        if s.content:
            t = s.temb(tok) + s.tpos[None]
            ctx.append(t * tokm[..., None].to(t.dtype))
            ctx.append((s.hpj(hp) + s.hpos[None]).flatten(1, 2))
        c = torch.cat(ctx, 1)
        for b in s.blocks:
            x = b(x, c)
        return s.head(s.nf(x))


def tweedie(lr, y, m, wlen, rho=1.5, ww=None):
    """lr log rate [B,NL,NE,W]; y window totals; m [B,W] valid; wlen [W] window lengths. mean deviance-like loss with each
    window weighted 1/sqrt(len) so near and far windows contribute comparably."""
    lmu = lr.float() + torch.log(wlen)
    l = -y * torch.exp(lmu * (1 - rho)) / (1 - rho) + torch.exp(lmu * (2 - rho)) / (2 - rho)
    w = m.float()[:, None, None, :] / wlen.sqrt()
    if ww is not None: w = w * ww                    # extra per-window weights (--wnear: tap uses [0, 64))
    return (l * w).sum() / (w.sum() * NL * NE + 1e-9)
