"""NestQuant v0 reference decoder (thread 12), fitted to the thread-04 kernel layout (nqk2.cu, REPORT2.md).

Dense fp32 reference decode of one expert at level 2 or 4 from the stored planes alone.

Geometry (rotated basis, kernel orientation W[N=out, K=in]; the encoder works on the EXL3 (k=in, n=out) transpose):
  unit      = one 16-row strip x 128-k chunk (2048 weights) = one kernel mma block = 8 tail-biting rings of 256
              weights (G=4 lanes share a ring). Ring g of a unit holds rows {g, g+8} x all 128 k; ring position
              p = t4*64 + j (lane 4g+t4, lane weight j) maps to row g + 8*(r&1), k 16t + 2*t4 + 8*(r>>1) + e with
              pp=j>>1, e=j&1, t=pp>>2, r=pp&3 (mma.m16n8k16 A-fragment order, nq2.ref_W).
  state     trellis state at ring position p = (sum_j sym[p+j] << K*j) & 0xFFFF (ring wraps), i.e. the 16-bit
              window starting at bit K*p of the ring's LSB-first bit stream (kernel's lane record + neighbour word).
  value     mul1: fp16(A*(1024 + bytesum(state*0x83DCD12D)) + B), A=0x1eee, B=0xc931 (harness.codebook_lut).
  delta     one fp16 per unit (16x128 block, kernel default) or, as a closer, one per ring.
  TP shard  256 intermediate channels: gate/up = 16 strips x all chunks, down = 2 chunks x all strips
              (768 units / shard either way). Units inside a shard are stored strip-major (strip, chunk).

Planes (per projection, per shard, constant size per (projection, plane, shard)):
  base  K_u-bit symbols of every unit (K_u = 2, or the MiMo 1/3 down split), + su2/sv2 fp16
  p4    residual low-2-bit symbols, delta, (optional) residual codebook id of every unit, + su4/sv4
  p4x   top bit of the 3-bit residual symbols of the K3 units (shard order) + the K3 unit mask
Level 2 reads base; level 4 reads base + p4 + p4x.  (Level 3 dropped by user decision 2026-09-28.)

Reconstruction:   Q2 = lut_mul1[state_base];   Q4 = Q2 + delta * lut_cb[state_res]
                  W_L = (had128_n(had128_k(Q_L.half()) * su_L) * sv_L)^T      (exllamav3 fp16 decode path)

CLI:  python nq_decode.py ARTIFACT.pt --level 4 [--out dense.pt] [--check-against internal.pt]
"""
import os, sys, math, argparse
os.environ.setdefault("OMP_NUM_THREADS", "8")
import torch

H05 = "/home/coder/git/nestquant/threads/05-exl3-harness"
if H05 not in sys.path:
    sys.path.insert(0, H05)

CODEBOOKS = ["mul1", "mcg", "3inst"]
_LUT, _RI = {}, {}


def lut(cb, device="cuda"):
    key = (cb, str(device))
    if key not in _LUT:
        import harness as h
        _LUT[key] = h.codebook_lut(cb, device)
    return _LUT[key]


def _Q():
    from exllamav3.modules.quant.exl3_lib import quantize as Q
    return Q


def ring_index(device="cuda"):
    """RI[g, p] = local index (k_local*16 + row_local) inside a [128 k, 16 n] unit of ring g, position p."""
    key = str(device)
    if key not in _RI:
        RI = torch.empty(8, 256, dtype=torch.long)
        for g in range(8):
            for t4 in range(4):
                for j in range(64):
                    pp, e = j >> 1, j & 1
                    t, r = pp >> 2, pp & 3
                    row = g + 8 * (r & 1)
                    kk = 16 * t + 2 * t4 + 8 * (r >> 1) + e
                    RI[g, t4 * 64 + j] = kk * 16 + row
        assert torch.equal(RI.flatten().sort().values, torch.arange(2048))
        _RI[key] = RI.to(device)
    return _RI[key]


# ------------------------------------------------------------------------------------------------ bits
def pack_bits(sym, K):
    """sym [T, 256] ints < 2^K -> uint8 [T, 32K], LSB-first bit stream (= kernel lane records, lanes in order)."""
    T = sym.shape[0]
    bits = (sym.long().unsqueeze(-1) >> torch.arange(K, device=sym.device)) & 1
    bits = bits.reshape(T, 32 * K, 8)
    return (bits << torch.arange(8, device=sym.device)).sum(-1).to(torch.uint8)


def unpack_bits(buf, K):
    T = buf.shape[0]
    b = (buf.long().unsqueeze(-1) >> torch.arange(8, device=buf.device)) & 1
    b = b.reshape(T, 256, K)
    return (b << torch.arange(K, device=buf.device)).sum(-1)


def states_from_symbols(sym, K):
    st = torch.zeros_like(sym)
    for j in range(math.ceil(16 / K)):
        st |= torch.roll(sym, -j, dims=1) << (K * j)
    return st & 0xFFFF


# ------------------------------------------------------------------------------------------------ layout
def unit_order(tk, tn, shard_axis, device="cpu"):
    """Flat unit ids u = a*tn + c (a = 128-k chunk, c = 16-n strip) in storage order: shard-major, then (c, a).
    Returns (order [tk*tn], shard_of_unit [tk*tn] in flat-id order)."""
    a = torch.arange(tk, device=device).repeat_interleave(tn)
    c = torch.arange(tn, device=device).repeat(tk)
    shard = c // 16 if shard_axis == "n" else a // 2
    key = (shard * tn + c) * tk + a
    return torch.argsort(key), shard


def superblock_of_unit(tk, tn, device="cpu"):
    a = torch.arange(tk, device=device).repeat_interleave(tn)
    c = torch.arange(tn, device=device).repeat(tk)
    return a * (tn // 8) + c // 8


def superblock_order(tk, tn, shard_axis, device="cpu"):
    """Flat superblock ids (a*(tn/8) + c8) in storage order (shard-major, then (c8, a)) and shard of each sb."""
    t8 = tn // 8
    a = torch.arange(tk, device=device).repeat_interleave(t8)
    c8 = torch.arange(t8, device=device).repeat(tk)
    shard = c8 // 2 if shard_axis == "n" else a // 2
    return torch.argsort((shard * t8 + c8) * tk + a), shard


# ------------------------------------------------------------------------------------------------ decode
BASE_VARIANTS = {"sign": [1.0, -1.0], "sg4": [s * g for g in (0.9, 0.97, 1.03, 1.1) for s in (1, -1)]}


def variant_table(name, device="cuda"):
    """Per-ring base scale variants, fp16-representable (the kernel folds a into its fp16 hfma constants)."""
    return torch.tensor(BASE_VARIANTS[name], device=device).half().float()


def variant_bits(name):
    return math.ceil(math.log2(len(BASE_VARIANTS[name])))


def _values(sym, K, cb, device):
    return lut(cb, device)[states_from_symbols(sym, K)]


@torch.no_grad()
def rotated_levels(P, device="cuda"):
    """Rotated-basis reconstructions {2, 4: [k, n] fp32} of one projection from its planes."""
    m = P["meta"]; k, n, tk, tn = m["k"], m["n"], m["tk"], m["tn"]
    U = tk * tn
    order, _ = unit_order(tk, tn, m["shard_axis"], device)
    RI = ring_index(device).flatten()
    Kb = torch.as_tensor(m["base_K"], device=device).view(tk, 1).expand(tk, tn).flatten()   # per chunk
    # ---- base: units back to back in storage order, 256*K bytes each
    raw = torch.cat([b.to(device) for b in P["base"]["shards"]])
    Ks = Kb[order]
    offs = torch.cat([torch.zeros(1, dtype=torch.long, device=device), torch.cumsum(Ks * 256, 0)])
    Q2 = torch.empty(U, 2048, device=device)
    for K in sorted(set(Ks.tolist())):
        sel = (Ks == K).nonzero().flatten()
        idx = offs[sel].unsqueeze(1) + torch.arange(256 * K, device=device)
        sym = unpack_bits(raw[idx].view(-1, 32 * K), K)                 # [8*len, 256]
        v = _values(sym, K, "mul1", device)
        if m.get("base_var"):
            var = torch.cat([b.to(device) for b in P["base"]["var"]]).long().view(U, 8)        # storage order
            v = v * variant_table(m["base_var"], device)[var[sel]].view(-1, 1)
        v = v.view(len(sel), 2048)
        Q2[order[sel].unsqueeze(1), RI.unsqueeze(0)] = v
    # ---- residual units
    k3 = torch.zeros(U, dtype=torch.bool, device=device)
    xsyms = torch.zeros(U, 8, 256, dtype=torch.long, device=device)
    k3_ids = order[torch.cat([s.to(device) for s in P["p4x"]["mask"]]).bool()]            # storage order
    if len(k3_ids):
        hi = unpack_bits(torch.cat([s.to(device) for s in P["p4x"]["shards"]]).view(-1, 32), 1)
        k3[k3_ids] = True
        xsyms[k3_ids] = hi.view(-1, 8, 256)
    pl = P["p4"]; uid = order
    lo = unpack_bits(torch.cat([s.to(device) for s in pl["shards"]]).view(-1, 64), 2).view(-1, 8, 256)
    cb = torch.cat([s.to(device) for s in pl["cb"]]).long() if pl["cb"] else torch.zeros(0, device=device)
    if cb.numel() == 0:
        cb = torch.zeros(len(uid), dtype=torch.long, device=device)
    dl = torch.cat([s.to(device) for s in pl["delta"]]).float().view(len(uid), -1)        # [units, 1 or 8]
    sym = lo | (xsyms[uid] << 2)
    g = torch.empty(len(uid), 8, 256, device=device)
    for K in (2, 3):
        for ci, cbn in enumerate(CODEBOOKS):
            sel = ((k3[uid] == (K == 3)) & (cb == ci)).nonzero().flatten()
            if len(sel):
                g[sel] = _values(sym[sel].view(-1, 256), K, cbn, device).view(-1, 8, 256)
    d = dl.unsqueeze(-1) if dl.shape[1] == 8 else dl.view(-1, 1, 1)
    r = torch.empty(len(uid), 2048, device=device)
    r[:, RI] = (d * g).view(len(uid), 2048)
    Q4 = Q2.clone(); Q4[uid] = Q2[uid] + r
    Qs = {2: Q2, 4: Q4}
    out = {}
    for L, Qu in Qs.items():
        out[L] = Qu.view(tk, tn, 128, 16).permute(0, 2, 1, 3).reshape(k, n).contiguous()
    return out


@torch.no_grad()
def dense_from_rotated(Q, suh, svh):
    """exllamav3 fp16 decode path (LinearEXL3.get_weight_tensor): returns [out, in] fp32."""
    Qm = _Q()
    w = Q.half()
    w = Qm.preapply_had_l(w, 128); w *= suh.to(w.device).unsqueeze(1)
    w = Qm.preapply_had_r(w, 128); w *= svh.to(w.device).unsqueeze(0)
    return w.float().T.contiguous()


SCALE_PLANE = {2: "base", 4: "p4"}


@torch.no_grad()
def decode_matrix(P, level, device="cuda", rot=None):
    rot = rot if rot is not None else rotated_levels(P, device)
    pl = P[SCALE_PLANE[level]]
    return dense_from_rotated(rot[level], pl["suh"].to(device), pl["svh"].to(device))


@torch.no_grad()
def decode_expert(art, level, device="cuda"):
    """art = torch.load(encoder artifact): {gate, up, down: planes, meta}. -> [g, u, d] fp32 [out, in]
    (un-permutes the intermediate channels if the artifact was fitted in act-order, e.g. the MiMo down split)."""
    W = [decode_matrix(art[p], level, device) for p in ("gate", "up", "down")]
    perm = art.get("meta", {}).get("inter_perm")
    if perm is not None:
        inv = torch.argsort(torch.as_tensor(perm, device=device))
        W = [W[0][inv], W[1][inv], W[2][:, inv]]
    return W


def plane_bytes(P):
    """Bytes per (plane, shard) for one projection (symbols + per-unit metadata + level scales, per shard)."""
    out = {}
    nsh = len(P["base"]["shards"])
    for name in ("base", "p4", "p4x"):
        pl = P[name]
        per = []
        for s in range(nsh):
            b = pl["shards"][s].numel()
            for key in ("delta", "cb", "mask", "var"):
                if key in pl:
                    if not pl[key]:
                        continue
                    t = pl[key][s]
                    if key == "var":
                        b += (t.numel() * variant_bits(P["meta"]["base_var"]) + 7) // 8; continue
                    b += (t.numel() + 7) // 8 if key == "mask" else (t.numel() + 3) // 4 if key == "cb" else t.numel() * t.element_size()
            per.append(b)
        sc = 0
        if "suh" in pl:     # scales: full input-side vector + the shard's output slice (gate/up); transposed for down
            m = P["meta"]
            sc = 2 * ((m["k"] + m["n"] // nsh) if m["shard_axis"] == "n" else (m["k"] // nsh + m["n"]))
        out[name] = dict(min=min(per), max=max(per), scales=sc)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact"); ap.add_argument("--level", type=int, default=4, choices=[2, 4])
    ap.add_argument("--out"); ap.add_argument("--check-against", help="encoder internal dense .pt {level: [g,u,d]}")
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(12 / 80)
    art = torch.load(a.artifact, weights_only=False, map_location="cpu")
    W = decode_expert(art, a.level)
    if a.out:
        torch.save({k: w.cpu() for k, w in zip(["gate", "up", "down"], W)}, a.out)
    if a.check_against:
        ref = torch.load(a.check_against, weights_only=False)[a.level]
        print("bit-exact:", all(torch.equal(x.cpu(), y.cpu()) for x, y in zip(W, ref)))


if __name__ == "__main__":
    main()
