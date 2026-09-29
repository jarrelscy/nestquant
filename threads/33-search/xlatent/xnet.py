"""T33b xlatent model: shared cross-layer latent z(t) + per-layer factorised decoders.
Per block b (end of block, serve-causal), per layer L, per expert e:
  own features f[b,L,e,:] (GPU EMA scan over past blocks, reset per chain; normalised like v2: salience / layer's EMA256
  salience-per-hit):  log1p of  sema32 sema128 sema512 sal16 ema32(cnt) ema128(cnt)  + log(S_v2) (v2 GBDT score)
  encoder: per-layer linear of [lsema128, lsema512] (512 -> r) -> concat 75r -> MLP -> z_in -> GRU / EMA latent -> z
  decoder: log mu = MLP_e(f, c_L(z)) + a_L log S_v2 + b_L[e] + U_L[e] . z        (mu in hits-units of next-64 sal)
"""
import numpy as np
import torch
import torch.nn as nn

NL, NE = 75, 256
G = 16
FEAT_NAMES = ("lsv2", "sema32", "sema128", "sema512", "sal16", "ema32", "ema128")
NC = len(FEAT_NAMES)


@torch.no_grad()
def own_features(bsal, bcnt, S_v2, sg, dev, out_dtype=torch.float16, chunk=4096):
    """bsal/bcnt/S_v2 array-likes [nb,75,256] -> torch [nb,75,256,NC] (on CPU, out_dtype).  Sequential scan on dev."""
    nb = bsal.shape[0]
    out = torch.empty((nb, NL, NE, NC), dtype=out_dtype)
    a = {h: 0.5 ** (G / h) for h in (32, 128, 256, 512)}
    starts = {s for s, e in sg}
    Es = {h: torch.zeros(NL, NE, device=dev, dtype=torch.float64) for h in a}
    Ec = {h: torch.zeros(NL, NE, device=dev, dtype=torch.float64) for h in a}
    for c0 in range(0, nb, chunk):
        c1 = min(nb, c0 + chunk)
        bs = torch.from_numpy(np.asarray(bsal[c0:c1], np.float32)).to(dev, torch.float64)
        bc = torch.from_numpy(np.asarray(bcnt[c0:c1], np.float32)).to(dev, torch.float64)
        sv = torch.from_numpy(np.asarray(S_v2[c0:c1], np.float32)).to(dev)
        o = torch.empty((c1 - c0, NL, NE, NC), device=dev, dtype=torch.float32)
        for j in range(c1 - c0):
            k = c0 + j
            if k in starts:
                for h in a:
                    Es[h].zero_(); Ec[h].zero_()
            for h, al in a.items():
                Es[h].mul_(al).add_(bs[j]); Ec[h].mul_(al).add_(bc[j])
            nrm = Es[256].sum(1) / Ec[256].sum(1).clamp_min(1e-30)
            nrm = torch.where(nrm > 0, nrm, torch.ones_like(nrm))[:, None]
            o[j, ..., 1] = (Es[32] * ((1 - a[32]) / G) / nrm).float()
            o[j, ..., 2] = (Es[128] * ((1 - a[128]) / G) / nrm).float()
            o[j, ..., 3] = (Es[512] * ((1 - a[512]) / G) / nrm).float()
            o[j, ..., 4] = (bs[j] / nrm).float()
            o[j, ..., 5] = (Ec[32] * ((1 - a[32]) / G)).float()
            o[j, ..., 6] = (Ec[128] * ((1 - a[128]) / G)).float()
        o[..., 1:] = torch.log1p(o[..., 1:] * 16)
        o[..., 0] = torch.log(sv.clamp_min(1e-3))
        out[c0:c1] = o.to(out_dtype).cpu()
    return out


class XLatent(nn.Module):
    def __init__(self, dz=32, r=8, hid=128, latent="gru", use_ctx=True, use_v2=True, mlp_hid=32, dz_zero=False):
        super().__init__()
        self.dz, self.latent, self.use_ctx, self.use_v2, self.dz_zero = dz, latent, use_ctx, use_v2, dz_zero
        self.enc_w = nn.Parameter(torch.randn(NL, 2 * NE, r) * (1.0 / np.sqrt(2 * NE)))
        self.enc_b = nn.Parameter(torch.zeros(NL, r))
        self.enc = nn.Sequential(nn.Linear(NL * r, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU())
        if latent == "gru":
            self.cell = nn.GRUCell(hid, dz)
        else:                                          # learned-decay EMA latent
            self.proj = nn.Linear(hid, dz)
            self.logit_decay = nn.Parameter(torch.linspace(0, 4, dz))
        nctx = 4 if use_ctx else 0
        self.ctx = nn.Parameter(torch.zeros(NL, dz, nctx)) if use_ctx else None
        nin = NC + nctx + NL * 0
        self.lemb = nn.Parameter(torch.zeros(NL, 4))
        self.mlp = nn.Sequential(nn.Linear(nin + 4, mlp_hid), nn.GELU(), nn.Linear(mlp_hid, mlp_hid), nn.GELU(),
                                 nn.Linear(mlp_hid, 1))
        self.a = nn.Parameter(torch.ones(NL) * (1.0 if use_v2 else 0.0))
        self.b = nn.Parameter(torch.zeros(NL, NE))
        self.U = nn.Parameter(torch.zeros(NL, NE, dz))

    def z_in(self, f):
        """f [B, T, NL, NE, NC] -> [B, T, hid]"""
        x = torch.cat([f[..., 2], f[..., 3]], -1)                   # [B,T,NL,512]
        h = torch.einsum("btlk,lkr->btlr", x, self.enc_w) + self.enc_b
        return self.enc(h.flatten(2))

    def step_latent(self, u, st):
        """u [B, hid] -> z [B, dz]"""
        if self.latent == "gru":
            return self.cell(u, st)
        d = torch.sigmoid(self.logit_decay)
        return st * d + (1 - d) * self.proj(u)

    def decode(self, f, z):
        """f [B,T,NL,NE,NC], z [B,T,dz] -> log mu [B,T,NL,NE]"""
        if self.dz_zero:
            z = torch.zeros_like(z)
        B, T_ = f.shape[:2]
        parts = [f.float()]
        if self.use_ctx:
            c = torch.einsum("btd,ldc->btlc", z, self.ctx)
            parts.append(c[:, :, :, None, :].expand(B, T_, NL, NE, c.shape[-1]))
        parts.append(self.lemb[None, None, :, None, :].expand(B, T_, NL, NE, 4))
        x = torch.cat(parts, -1)
        base = self.mlp(x)[..., 0]
        out = base + self.b + torch.einsum("btd,led->btle", z, self.U)
        if self.use_v2:
            out = out + self.a[:, None] * f[..., 0].float()
        return out

    def forward(self, f, st):
        """f [B,T,...] chunk; st [B,dz] -> logmu [B,T,NL,NE], new st"""
        u = self.z_in(f.float())
        zs = []
        for t in range(u.shape[1]):
            st = self.step_latent(u[:, t], st)
            zs.append(st)
        z = torch.stack(zs, 1)
        return self.decode(f, z), st, z


def tweedie_loss(logmu, y, m, p=1.5):
    mu1 = torch.exp(logmu * (1 - p)); mu2 = torch.exp(logmu * (2 - p))
    l = -y * mu1 / (1 - p) + mu2 / (2 - p)
    return (l * m).sum() / m.sum()
