"""NestQuant v0 reference decoder (thread 12).

Dense fp32 reference decode of one expert at level 2, 3 or 4 from the stored planes alone.

Planes (per projection; see nq_encode.py for how they are fitted):
    base   K_a-bit mul1 trellis symbols for every 16x16 tile (K per 16-row input block a, usually 2),
           tiles in shard-major order, plus su2/sv2 fp16 scales.
    p4a    level-3 tile subset: 2-bit symbols (low bits for K=3 residual tiles), residual codebook id,
           per-tile fp16 delta, the level-3 tile mask, and su3/sv3.
    p4b    remaining tiles' residual symbols / codebook ids / deltas, and su4/sv4.
    p4x    top bit of the 3-bit residual symbols for the K3 tiles (all of them lie in p4a), and the K3 mask.
Level 2 reads base; level 3 reads base + p4a (+ p4x); level 4 reads base + p4a + p4b (+ p4x).
Each level uses its own (refit) output/input scales, which live in the level's plane.

Reconstruction in the rotated basis (tile t = (a, c), trellis order inside the tile):
    Q2[t]  = lut_mul1[state_base]
    Q_L[t] = Q2[t] + delta[t] * lut_cb[t][state_res]         for tiles present at level L, else Q2[t]
    W_L    = (had128_n( had128_k(Q_L.half()) * su_L ) * sv_L)^T        (exllamav3 fp16 decode path)
The trellis state at step i is (sum_j sym[i-j] << K*j) & 0xFFFF, indices wrap around the 256-step ring.

CLI:  python nq_decode.py EXPERT.pt --level 4 [--out dense.pt] [--check-against internal.pt]
"""
import os, sys, math, argparse
os.environ.setdefault("OMP_NUM_THREADS", "8")
import torch

H05 = "/home/coder/git/nestquant/threads/05-exl3-harness"
if H05 not in sys.path:
    sys.path.insert(0, H05)

_LUT = {}


def lut(cb, device="cuda"):
    if cb not in _LUT:
        import harness as h
        _LUT[cb] = h.codebook_lut(cb, device)
    return _LUT[cb]


CODEBOOKS = ["mul1", "mcg", "3inst"]


def _Q():
    from exllamav3.modules.quant.exl3_lib import quantize as Q
    return Q


def tc_perm(device="cuda"):
    return _Q().tensor_core_perm(device).long()


# ------------------------------------------------------------------------------------------------ bits
def pack_bits(sym, K):
    """sym: [T, 256] integer symbols < 2^K  ->  uint8 [T, 32*K] (LSB-first bit stream per tile)."""
    T = sym.shape[0]
    s = sym.long()
    bits = (s.unsqueeze(-1) >> torch.arange(K, device=s.device)) & 1          # [T, 256, K]
    bits = bits.reshape(T, 32 * K, 8)
    return (bits << torch.arange(8, device=s.device)).sum(-1).to(torch.uint8)


def unpack_bits(buf, K):
    T = buf.shape[0]
    b = (buf.long().unsqueeze(-1) >> torch.arange(8, device=buf.device)) & 1  # [T, 32K, 8]
    b = b.reshape(T, 256, K)
    return (b << torch.arange(K, device=buf.device)).sum(-1)


def states_from_symbols(sym, K):
    """Tail-biting shift-register states of a 256-step ring: state_i = (sum_j sym_{i-j} << K j) & 0xFFFF."""
    st = torch.zeros_like(sym)
    for j in range(math.ceil(16 / K)):
        st |= torch.roll(sym, j, dims=1) << (K * j)
    return st & 0xFFFF


# ------------------------------------------------------------------------------------------------ layout
def shard_order(tk, tn, shard_axis, shard_tiles=16, device="cpu"):
    """Flat tile ids (a*tn + c) in shard-major order: shard s covers tile columns (axis 'n') or tile rows
    (axis 'k') [16s, 16s+16), i.e. 256 intermediate channels; inside a shard tiles run in (a, c) order."""
    a = torch.arange(tk, device=device).repeat_interleave(tn)
    c = torch.arange(tn, device=device).repeat(tk)
    shard = (c if shard_axis == "n" else a) // shard_tiles
    key = shard * (tk * tn) + a * tn + c
    return torch.argsort(key)


def base_K_per_tile(meta, device="cpu"):
    Ka = torch.tensor(meta["base_K"], device=device)
    return Ka.repeat_interleave(meta["tn"])            # flat (a, c) order


# ------------------------------------------------------------------------------------------------ decode
@torch.no_grad()
def rotated_levels(P, device="cuda"):
    """Rotated-basis reconstructions {2: Q2, 3: Q3, 4: Q4} ([k, n] fp32) from the planes of one projection."""
    m = P["meta"]; k, n, tk, tn = m["k"], m["n"], m["tk"], m["tn"]
    perm = tc_perm(device)
    order = shard_order(tk, tn, m["shard_axis"], device=device)
    Kt = base_K_per_tile(m, device)[order]            # base K of each tile in stored order
    raw = P["base"]["sym"].to(device)
    vals = torch.empty((tk * tn, 256), device=device)
    off = 0
    # base tiles are stored back to back with K*32 bytes each
    offs = torch.cat([torch.zeros(1, dtype=torch.long, device=device), torch.cumsum(Kt * 32, 0)])
    for K in sorted(set(Kt.tolist())):
        sel = (Kt == K).nonzero().flatten()
        idx = offs[sel].unsqueeze(1) + torch.arange(32 * K, device=device)
        sym = unpack_bits(raw[idx], K)
        vals[order[sel]] = lut("mul1", device)[states_from_symbols(sym, K)]
    Q2t = vals                                          # [tiles, 256] trellis order, flat (a, c)

    def residual(plane, ids):
        """ids: flat tile ids (stored order) present in `plane`; returns delta*g tiles in trellis order."""
        sym = unpack_bits(plane["sym"].to(device), 2)
        if "x_rows" in plane:                           # 3-bit tiles: top bit from p4x
            xr = plane["x_rows"].to(device)             # positions in this plane that are K3
            hi = unpack_bits(P["p4x"]["sym"].to(device), 1)[plane["x_src"].to(device)]
            sym[xr] = sym[xr] | (hi << 2)
            Kt_ = torch.full((len(ids),), 2, device=device, dtype=torch.long); Kt_[xr] = 3
        else:
            Kt_ = torch.full((len(ids),), 2, device=device, dtype=torch.long)
        cb = plane["cb"].to(device).long()
        g = torch.empty((len(ids), 256), device=device)
        for K in (2, 3):
            for ci, name in enumerate(CODEBOOKS):
                sel = ((Kt_ == K) & (cb == ci)).nonzero().flatten()
                if len(sel):
                    g[sel] = lut(name, device)[states_from_symbols(sym[sel], K)]
        d = delta_values(P, plane, ids, device)
        return d.unsqueeze(1) * g

    out = {2: Q2t}
    ids_a = P["p4a"]["ids"].to(device).long(); ids_b = P["p4b"]["ids"].to(device).long()
    Ra = residual(P["p4a"], ids_a); Rb = residual(P["p4b"], ids_b)
    Q3 = Q2t.clone(); Q3[ids_a] = Q2t[ids_a] + Ra
    Q4 = Q3.clone(); Q4[ids_b] = Q2t[ids_b] + Rb
    out[3], out[4] = Q3, Q4
    res = {}
    pi = torch.argsort(perm)
    for L, Qt in out.items():
        nat = Qt[:, pi].view(tk, tn, 16, 16).permute(0, 2, 1, 3).reshape(k, n)
        res[L] = nat.contiguous()
    return res


def delta_values(P, plane, ids, device):
    m = P["meta"]
    if m["delta_mode"] == "tile":
        return plane["delta"].to(device).float()
    # rank-1 fp16 map U[a] * V[c] rounded to fp16, sign per tile
    U = P["p4a"]["U"].to(device).float(); V = P["p4a"]["V"].to(device).float()
    a = ids // m["tn"]; c = ids % m["tn"]
    d = (U[a] * V[c]).half().float()
    return torch.where(plane["neg"].to(device).bool(), -d, d)


@torch.no_grad()
def dense_from_rotated(Q, suh, svh):
    """exllamav3 fp16 decode path (LinearEXL3.get_weight_tensor): returns [out, in] fp32."""
    Qm = _Q()
    w = Q.half()
    w = Qm.preapply_had_l(w, 128); w *= suh.to(w.device).unsqueeze(1)
    w = Qm.preapply_had_r(w, 128); w *= svh.to(w.device).unsqueeze(0)
    return w.float().T.contiguous()


def level_scales(P, level):
    key = {2: "base", 3: "p4a", 4: "p4b"}[level]
    return P[key]["suh"], P[key]["svh"]


@torch.no_grad()
def decode_matrix(P, level, device="cuda", rot=None):
    rot = rot if rot is not None else rotated_levels(P, device)
    suh, svh = level_scales(P, level)
    W = dense_from_rotated(rot[level], suh, svh)
    m = P["meta"]
    if m.get("out_perm") is not None:          # MiMo act-order: undo the intermediate-channel permutation
        inv = torch.argsort(torch.as_tensor(m["out_perm"], device=W.device))
        W = W[inv] if m["perm_axis"] == "out" else W[:, inv]
    return W


@torch.no_grad()
def decode_expert(art, level, device="cuda"):
    """art: dict {gate, up, down: planes} (torch.load of an encoder artifact) -> [g, u, d] fp32 [out, in]."""
    return [decode_matrix(art[p], level, device) for p in ("gate", "up", "down")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact"); ap.add_argument("--level", type=int, default=4, choices=[2, 3, 4])
    ap.add_argument("--out"); ap.add_argument("--check-against", help="encoder's internal dense .pt {level: [g,u,d]}")
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(12 / 80)
    art = torch.load(a.artifact, weights_only=False)
    W = decode_expert(art, a.level)
    if a.out:
        torch.save({k: w.cpu() for k, w in zip(["gate", "up", "down"], W)}, a.out)
    if a.check_against:
        ref = torch.load(a.check_against, weights_only=False)[a.level]
        print("bit-exact:", all(torch.equal(x.cpu(), y.cpu()) for x, y in zip(W, ref)))


if __name__ == "__main__":
    main()
