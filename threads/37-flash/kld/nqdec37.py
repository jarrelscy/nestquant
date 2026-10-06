"""T37 KLD: standalone NestQuant v1 expert decoder (dense bf16 at level 2 / 4) for the HF-forward KLD harness.

Same arithmetic as the production reference decoder (threads/12 nq_decode.decode_expert with the T35 nq15
pattern-rate base, i.e. ring_levels reading the base streams at meta base_K), but importable from /tmp/venv-t37g:
  - nq_decode is imported for its pure-torch stream / fold / layout functions (module import has no exllamav3 dep);
  - nq15.ring_levels is re-stated here (nq15 itself imports the encoder + harness + exllamav3);
  - exllamav3's dense_from_rotated (fp16 Hadamard-128 l/r + suh/svh) uses the exported exllamav3 Hadamard
    matrix had128.pt (get_hadamard_dt(128, fp32, 1/sqrt(128)), exported from the glm52 venv) -- identical math.
Validated against the official decode (validate_decode.py): bitwise fp32 equality expected.

  W = decode_expert(art, (2, 4), device)  -> {2: (gu [2F, D] bf16, dn [D, F] bf16), 4: (...)}
"""
import os
import sys

import torch

T12 = "/home/coder/git/nestquant/threads/12-reference-encoder"
if T12 not in sys.path:
    sys.path.insert(0, T12)
import nq_decode as D  # noqa: E402

HAD = os.environ.get("NQ37_HAD128", "/tmp/nestquant/37-flash/kld/had128.pt")


def bres(n):
    return sum(1 << j for j in range(16) if ((j + 1) * n) // 16 > (j * n) // 16)


for _K, _v in {1.25: (1, 0x8888), 2.5625: (2, bres(9)), 2.8125: (2, bres(13))}.items():   # = nq15.NEW
    D.PATTERNS.setdefault(_K, _v)
assert bres(5) == 0x9248 and D.PATTERNS[1.5] == (1, 0xAAAA)
import functools  # noqa: E402
D.ring_bytes = functools.lru_cache(maxsize=None)(D.ring_bytes)     # pure in K; was ~1 s/expert of per-unit host work

_H = {}


def had(device):
    k = str(device)
    if k not in _H:
        _H[k] = torch.load(HAD, map_location="cpu", weights_only=True).float().to(device)
    return _H[k]


@torch.no_grad()
def ring_levels(P, device):
    """== nq15.ring_levels (base streams at meta base_K)."""
    m = P["meta"]; tk, tn = m["tk"], m["tn"]
    order, _ = D.unit_order(tk, tn, m["shard_axis"], device)
    U = tk * tn
    sb = D._gather_streams(D._cat(P["base"]["shards"], device), torch.full((U,), float(m.get("base_K", 2)), device=device),
                           device)
    Sb = D.hsum(sb)
    a = None
    if m.get("base_var"):
        var = D._cat(P["base"]["var"], device).long().view(U, 8)
        a = D.variant_table(m["base_var"], device)[var].unsqueeze(-1)
    Q2 = D.q2_values(Sb, a)
    mflat = None
    if P["p4"].get("mask"):
        mflat = torch.zeros(U, dtype=torch.bool, device=device)
        mflat[order] = D._cat(P["p4"]["mask"], device).bool()
    Ku = D.res_K_units(m, mflat, device)[order]
    sr = D._gather_streams(D._cat(P["p4"]["shards"], device), Ku, device)
    Sr = D.hsum(sr)
    bw = D._cat(P["p4"]["word"], device).long()
    Mb, N = bw & 255, (bw >> 8) & 255
    Q4 = D.fold(Sb, Sr, Mb, N, a)
    return dict(Q2=Q2, Q4=Q4, order=order)


@torch.no_grad()
def _had_l(x):                                   # exllamav3 preapply_had_l (fp32 math, back to x dtype)
    k, n = x.shape
    return (had(x.device) @ x.float().view(-1, 128, n)).view(k, n).to(x.dtype)


@torch.no_grad()
def _had_r(x):
    k, n = x.shape
    return (x.float().view(k, -1, 128) @ had(x.device)).view(k, n).to(x.dtype)


@torch.no_grad()
def dense_from_rotated(Q, suh, svh):
    """== nq_decode.dense_from_rotated (exllamav3 LinearEXL3.get_weight_tensor fp16 path) -> [out, in] fp32."""
    w = Q.half()
    w = _had_l(w); w *= suh.to(w.device).unsqueeze(1)
    w = _had_r(w); w *= svh.to(w.device).unsqueeze(0)
    return w.float().T.contiguous()


@torch.no_grad()
def decode_matrix_levels(P, levels, device):
    m = P["meta"]
    rl = ring_levels(P, device)
    out = {}
    for L in levels:
        Q = D.to_matrix(rl[f"Q{L}"], rl["order"], m["tk"], m["tn"], m["k"], m["n"], device)
        pl = P[D.SCALE_PLANE[L]]
        W = dense_from_rotated(Q, pl["suh"].to(device), pl["svh"].to(device))
        out[L] = D.apply_ocol(P, W, L)
    return out


@torch.no_grad()
def decode_expert_f32(art, levels=(2, 4), device="cpu"):
    """-> {level: [g, u, d] fp32 [out, in]} (== nq_decode.decode_expert per level, incl. inter_perm)."""
    per = {p: decode_matrix_levels(art[p], levels, device) for p in ("gate", "up", "down")}
    out = {}
    perm = art.get("meta", {}).get("inter_perm")
    for L in levels:
        W = [per[p][L] for p in ("gate", "up", "down")]
        if perm is not None:
            inv = torch.argsort(torch.as_tensor(perm, device=device))
            W = [W[0][inv], W[1][inv], W[2][:, inv]]
        out[L] = W
    return out


@torch.no_grad()
def decode_expert(art, levels=(2, 4), device="cpu"):
    """-> {level: (gu [2F, D] bf16 = cat(gate, up), dn [D, F] bf16)} (the HF Experts / capture37 layout)."""
    f = decode_expert_f32(art, levels, device)
    return {L: (torch.cat([W[0], W[1]], 0).bfloat16(), W[2].bfloat16()) for L, W in f.items()}
