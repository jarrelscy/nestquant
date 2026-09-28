"""NestQuant v1 reference decoder (thread 12): thread-04 kernel layout + thread-15 level-4 int-fold (ref15_spec).

Dense fp32 reference decode of one expert at level 2 or 4 from the stored planes alone (no level 3, user decision).

Geometry (rotated basis, kernel orientation W[N=out, K=in]; the encoder works on the EXL3 (k=in, n=out) transpose):
  unit      one 16-row strip x 128-k chunk (2048 weights) = one kernel mma block = 8 tail-biting rings of 256 weights
            (G=4 lanes / ring). Ring g, position p = t4*64 + j (lane 4g+t4, lane weight j) <-> row g + 8*(r&1),
            k 16t + 2*t4 + 8*(r>>1) + e, pp=j>>1, e=j&1, t=pp>>2, r=pp&3 (== ref15_spec.ring_index).
  stream    LSB-first bit stream per ring; widths w_p = KA + ((MASK >> (p%16)) & 1), offsets step_off(p);
            state(p) = ring bits [step_off(p), +16) with wrap (tail-biting).  K: 2=(2,0) 2.5=(2,0xAAAA) 3=(3,0) ...
  hash      S(state) = bytesum(state * 0x83DCD12D mod 2^32)
  LEVEL 2   Q2 = fp16(A*(1024 + S(sb)) + B) (== harness mul1 LUT), A = fp16 0x1eee, B = fp16 0xc931.
            base variant a per ring (thread 17 "sg4", optional):  Q2 = fp16(fp16(a*A)*(1024+S) + fp16(a*B))
  LEVEL 4   ref15 RM_P int fold with block word Mb | N<<8 per unit:
            F = (Mb*S(sb) + N*S(sr) + 128) >> 8,  A' = fp16(fp32(1.732421875) * fp32(1/Mb)),
            C = fp16(fp32(fp32(N*(1/Mb)) * K0 + K0) - 1024*A'),  Q4 = fp16(A'*(1024 + F) + C)   (== ref15_spec.fold)
            with base variant a:  Q4 = fp16(fp16(a*A')*(1024+F) + fp16(a*C))   (a folds into the final HFMA2 constants)
  residual K per unit from a rule in meta (no map): uniform K, or positional (the first `frac` of each TP shard's
            units in (chunk, strip) order = the LAST-processed chunks in LDL order get K_hi), or a stored unit mask.
  TP shard  256 intermediate channels: gate/up = 16 strips x all chunks, down = 2 chunks x all strips (768 units).
            Units inside a shard are stored chunk-strip order key (shard, strip, chunk).

Planes (per projection, per shard):
  base  ring streams of every unit (K 2), [sg4: 3-bit variant id per ring], + su2/sv2 fp16      <- identical for all rates
  p4    residual ring streams (K_u per unit), block words u16 (Mb | N<<8) per unit, [mask], + su4/sv4 fp16
Level 2 reads base; level 4 reads base + p4.

CLI:  python nq_decode.py ARTIFACT.pt --level 4 [--out dense.pt] [--check-against internal.pt] [--xcheck-ref15 N]
"""
import os, sys, math, argparse
os.environ.setdefault("OMP_NUM_THREADS", "8")
import torch

H05 = "/home/coder/git/nestquant/threads/05-exl3-harness"
if H05 not in sys.path:
    sys.path.insert(0, H05)

_RI = {}
A = float(torch.tensor([0x1eee], dtype=torch.int16).view(torch.float16)[0])
B = float(torch.tensor([0xc931 - 65536], dtype=torch.int16).view(torch.float16)[0])
K0 = 1024 * A + B                                             # -3.453125
PATTERNS = {1.5: (1, 0xAAAA), 1.75: (1, 0xEEEE), 1.875: (1, 0xFEFE), 1.9375: (1, 0xFFFE), 2: (2, 0),
            2.25: (2, 0x8888), 2.3125: (2, 0x9248), 2.5: (2, 0xAAAA), 2.75: (2, 0xEEEE), 3: (3, 0), 4: (4, 0)}
BASE_VARIANTS = {"sign": [1.0, -1.0], "sg4": [s * g for g in (0.9, 0.97, 1.03, 1.1) for s in (1, -1)]}


def f16(x):
    return x.double().half().double()


def variant_table(name, device="cuda"):
    return torch.tensor(BASE_VARIANTS[name], device=device, dtype=torch.float64).half().double()


def variant_bits(name):
    return math.ceil(math.log2(len(BASE_VARIANTS[name])))


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
                    RI[g, t4 * 64 + j] = (16 * t + 2 * t4 + 8 * (r >> 1) + e) * 16 + g + 8 * (r & 1)
        assert torch.equal(RI.flatten().sort().values, torch.arange(2048))
        _RI[key] = RI.to(device)
    return _RI[key]


# ------------------------------------------------------------------------------------------------ streams
def widths(K, device="cuda"):
    KA, MASK = PATTERNS[K]
    p = torch.arange(256, device=device)
    w = KA + ((MASK >> (p % 16)) & 1)
    off = torch.cumsum(w, 0) - w
    return w, off, int(w.sum())


def ring_bytes(K):
    return widths(K, "cpu")[2] // 8


def pack_stream(sym, K):
    """sym [R, 256] (kernel order, sym_p < 2^w_p) -> uint8 [R, nbits/8] LSB-first."""
    w, off, nb = widths(K, sym.device)
    R = sym.shape[0]
    bits = torch.zeros(R, nb, dtype=torch.long, device=sym.device)
    for b in range(int(w.max())):
        sel = (w > b).nonzero().flatten()
        bits[:, off[sel] + b] = (sym[:, sel].long() >> b) & 1
    return (bits.view(R, nb // 8, 8) << torch.arange(8, device=sym.device)).sum(-1).to(torch.uint8)


def stream_states(buf, K):
    """uint8 [R, nbits/8] -> 16-bit states [R, 256] at every ring position (tail-biting windows)."""
    w, off, nb = widths(K, buf.device)
    R = buf.shape[0]
    bits = ((buf.long().unsqueeze(-1) >> torch.arange(8, device=buf.device)) & 1).view(R, nb)
    idx = (off.unsqueeze(1) + torch.arange(16, device=buf.device)) % nb
    return (bits[:, idx] << torch.arange(16, device=buf.device)).sum(-1)


def symbols(st, K):
    w, _, _ = widths(K, st.device)
    return st & ((1 << w) - 1)


def hsum(st):
    x = (st.long() * 0x83DCD12D) & 0xFFFFFFFF
    return (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + ((x >> 24) & 255)


# ------------------------------------------------------------------------------------------------ values
def q2_values(S, a=None):
    """S [.., 256] int; a broadcastable per-ring variant (float64 fp16 values) or None."""
    if a is None:
        return f16(A * (1024 + S.double()) + B)
    Aa = f16(a * A); Ba = f16(a * B)
    return f16(Aa * (1024 + S.double()) + Ba)


def fold_consts(Mb, N):
    rcp = torch.ones_like(Mb, dtype=torch.float32) / Mb.float()
    Ah = f16(torch.tensor(1.732421875, dtype=torch.float32, device=Mb.device) * rcp)
    t = (N.float() * rcp) * torch.tensor(K0, dtype=torch.float32, device=Mb.device) + torch.tensor(K0, dtype=torch.float32, device=Mb.device)
    Ch = f16(t.double() - 1024 * Ah)
    return Ah, Ch


def fold(Sb, Sr, Mb, N, a=None):
    """Sb, Sr [U, 8, 256] int; Mb, N [U] int; a [U, 8, 1] or None -> Q4 [U, 8, 256] float64 (fp16 values)."""
    Mb_, N_ = Mb.view(-1, 1, 1).long(), N.view(-1, 1, 1).long()
    F = (Mb_ * Sb + N_ * Sr + 128) >> 8
    Ah, Ch = fold_consts(Mb_, N_)
    if a is not None:
        Ah = f16(a * Ah); Ch = f16(a * Ch)
    return f16(Ah * (1024 + F.double()) + Ch)


def delta_to_MbN(delta):
    """ref15: largest-Mb rational N/Mb nearest to delta >= 0, Mb + N <= 257, Mb <= 255."""
    d = delta.double().clamp_min(0)
    Mb = torch.clamp(torch.floor(257 / (1 + d)), max=255).long().clamp_min(1)
    N = torch.minimum(torch.round(d * Mb).long(), 257 - Mb)
    return Mb, N


# ------------------------------------------------------------------------------------------------ layout
def unit_order(tk, tn, shard_axis, device="cpu"):
    """Flat unit ids u = a*tn + c (a = 128-k chunk, c = 16-n strip) in storage order: shard-major, then (c, a).
    Returns (order [tk*tn], shard_of_unit [tk*tn] in flat-id order)."""
    a = torch.arange(tk, device=device).repeat_interleave(tn)
    c = torch.arange(tn, device=device).repeat(tk)
    shard = c // 16 if shard_axis == "n" else a // 2
    key = (shard * tn + c) * tk + a
    return torch.argsort(key), shard


def res_K_units(meta, mask=None, device="cpu"):
    """Residual K per flat unit [tk*tn] (float) from the meta rule (+ stored mask for kind 'mask')."""
    tk, tn = meta["tk"], meta["tn"]
    rule = meta["res_rule"]
    Ku = torch.full((tk * tn,), float(rule["K"]), device=device)
    if rule["kind"] == "uniform":
        return Ku
    if rule["kind"] == "mask":
        Ku[mask.to(device).bool()] = float(rule["K_hi"])
        return Ku
    assert rule["kind"] == "pos"
    _, shard = unit_order(tk, tn, meta["shard_axis"], device)
    a = torch.arange(tk, device=device).repeat_interleave(tn)
    c = torch.arange(tn, device=device).repeat(tk)
    for s in shard.unique().tolist():
        ids = (shard == s).nonzero().flatten()
        ids = ids[torch.argsort(a[ids] * tn + c[ids])]           # last-processed (lowest chunk) first
        Ku[ids[:int(round(rule["frac"] * len(ids)))]] = float(rule["K_hi"])
    return Ku


def _cat(lst, device):
    return torch.cat([t.to(device) for t in lst]) if len(lst) else torch.zeros(0, device=device)


# ------------------------------------------------------------------------------------------------ decode
def _gather_streams(raw, Ks, device):
    """raw uint8 (units back to back, 8 rings each, K_u per unit, storage order); Ks [U] -> states [U, 8, 256]."""
    U = len(Ks)
    per = torch.tensor([8 * ring_bytes(float(k)) for k in Ks.tolist()], device=device)
    offs = torch.cumsum(per, 0) - per
    assert int(per.sum()) == raw.numel(), (int(per.sum()), raw.numel())
    st = torch.empty(U, 8, 256, dtype=torch.long, device=device)
    for K in sorted(set(Ks.tolist())):
        sel = (Ks == K).nonzero().flatten()
        nbytes = 8 * ring_bytes(K)
        idx = offs[sel].unsqueeze(1) + torch.arange(nbytes, device=device)
        st[sel] = stream_states(raw[idx].view(-1, nbytes // 8), K).view(len(sel), 8, 256)
    return st


@torch.no_grad()
def ring_levels(P, device="cuda"):
    """-> dict(Sb, Sr [U,8,256] storage order, a [U,8,1] or None, Mb, N [U], Q2, Q4 ring-order float64, order)."""
    m = P["meta"]; tk, tn = m["tk"], m["tn"]
    order, _ = unit_order(tk, tn, m["shard_axis"], device)
    U = tk * tn
    sb = _gather_streams(_cat(P["base"]["shards"], device), torch.full((U,), 2.0, device=device), device)
    Sb = hsum(sb)
    a = None
    if m.get("base_var"):
        var = _cat(P["base"]["var"], device).long().view(U, 8)
        a = variant_table(m["base_var"], device)[var].unsqueeze(-1)
    Q2 = q2_values(Sb, a)
    mflat = None
    if P["p4"].get("mask"):                                          # stored in storage order -> flat unit ids
        mflat = torch.zeros(U, dtype=torch.bool, device=device)
        mflat[order] = _cat(P["p4"]["mask"], device).bool()
    Ku = res_K_units(m, mflat, device)[order]
    sr = _gather_streams(_cat(P["p4"]["shards"], device), Ku, device)
    Sr = hsum(sr)
    bw = _cat(P["p4"]["word"], device).long()
    Mb, N = bw & 255, (bw >> 8) & 255
    Q4 = fold(Sb, Sr, Mb, N, a)
    return dict(Sb=Sb, Sr=Sr, a=a, Mb=Mb, N=N, Q2=Q2, Q4=Q4, order=order, Ku=Ku)


def to_matrix(Qring, order, tk, tn, k, n, device="cuda"):
    """ring-order [U, 8, 256] (storage order) -> [k, n] fp32 rotated-basis matrix."""
    RI = ring_index(device).flatten()
    X = torch.empty(tk * tn, 2048, device=device)
    X[order.unsqueeze(1), RI.unsqueeze(0)] = Qring.reshape(len(order), 2048).float()
    return X.view(tk, tn, 128, 16).permute(0, 2, 1, 3).reshape(k, n).contiguous()


@torch.no_grad()
def rotated_levels(P, device="cuda"):
    m = P["meta"]
    rl = ring_levels(P, device)
    return {L: to_matrix(rl[f"Q{L}"], rl["order"], m["tk"], m["tn"], m["k"], m["n"], device) for L in (2, 4)}


@torch.no_grad()
def dense_from_rotated(Q, suh, svh):
    """exllamav3 fp16 decode path (LinearEXL3.get_weight_tensor): returns [out, in] fp32."""
    from exllamav3.modules.quant.exl3_lib import quantize as Qm
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
    """art = {gate, up, down: planes, [meta]} -> [g, u, d] fp32 [out, in] (un-permutes inter_perm if present)."""
    W = [decode_matrix(art[p], level, device) for p in ("gate", "up", "down")]
    perm = art.get("meta", {}).get("inter_perm")
    if perm is not None:
        inv = torch.argsort(torch.as_tensor(perm, device=device))
        W = [W[0][inv], W[1][inv], W[2][:, inv]]
    return W


def plane_bytes(P):
    """Bytes per (plane, shard) for one projection (+ fp16 scale bytes per shard: input vector + output slice)."""
    m = P["meta"]; nsh = len(P["base"]["shards"])
    out = {}
    for name in ("base", "p4"):
        pl = P[name]; per = []
        for s in range(nsh):
            b = pl["shards"][s].numel()
            if name == "base" and pl.get("var"):
                b += (pl["var"][s].numel() * variant_bits(m["base_var"]) + 7) // 8
            if name == "p4":
                b += 2 * pl["word"][s].numel()
                if pl.get("mask"):
                    b += (pl["mask"][s].numel() + 7) // 8
            per.append(b)
        sc = 2 * ((m["k"] + m["n"] // nsh) if m["shard_axis"] == "n" else (m["k"] // nsh + m["n"]))
        out[name] = dict(min=min(per), max=max(per), scales=sc)
    return out


def bits_per_level(P):
    m = P["meta"]; nw = m["k"] * m["n"]
    base = 8 * sum(t.numel() for t in P["base"]["shards"])
    if P["base"].get("var"):
        base += variant_bits(m["base_var"]) * sum(t.numel() for t in P["base"]["var"])
    p4 = 8 * sum(t.numel() for t in P["p4"]["shards"]) + 16 * sum(t.numel() for t in P["p4"]["word"])
    if P["p4"].get("mask"):
        p4 += sum(t.numel() for t in P["p4"]["mask"])
    sc = 16 * (m["k"] + m["n"])
    return {2: (base + sc) / nw, 4: (base + p4 + sc) / nw, "p4_only": (p4 + sc) / nw, "artifact": (base + p4 + 2 * sc) / nw}


@torch.no_grad()
def xcheck_ref15(P, nunits=16, seed=0):
    """Decode `nunits` random units with thread 15's numpy ref15_spec.decode_unit and compare with ring_levels.
    Per-ring sign variants (a = +-1) are checked as a * ref15 (fp16 negation is exact: the sign folds into the
    per-lane A'/C HFMA2 constants); other gain variants are not ref15 values and are skipped."""
    sys.path.insert(0, "/home/coder/git/nestquant/threads/15-level4-decode")
    import numpy as np, ref15_spec as R15
    m = P["meta"]
    rl = ring_levels(P, "cpu")
    U = m["tk"] * m["tn"]
    base = _cat(P["base"]["shards"], "cpu"); res = _cat(P["p4"]["shards"], "cpu")
    Ks = rl["Ku"]
    bper = torch.full((U,), 8 * ring_bytes(2), dtype=torch.long)
    rper = torch.tensor([8 * ring_bytes(float(k)) for k in Ks.tolist()])
    boff = torch.cumsum(bper, 0) - bper; roff = torch.cumsum(rper, 0) - rper
    g = torch.Generator().manual_seed(seed)
    bad = 0
    for u in torch.randint(0, U, (nunits,), generator=g).tolist():
        bs = base[boff[u]:boff[u] + bper[u]].view(8, -1).numpy()
        rs = res[roff[u]:roff[u] + rper[u]].view(8, -1).numpy()
        q2, q4 = R15.decode_unit(bs, rs, int(rl["Mb"][u]), int(rl["N"][u]), Kb=2, Kr=float(Ks[u]) if float(Ks[u]) % 1 else int(Ks[u]))
        if rl["a"] is not None:
            if m["base_var"] != "sign":
                return None
            a = rl["a"][u].numpy()
            q2, q4 = a * q2, a * q4
        bad += int((q2 != rl["Q2"][u].numpy()).sum() + (q4 != rl["Q4"][u].numpy()).sum())
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact"); ap.add_argument("--level", type=int, default=4, choices=[2, 4])
    ap.add_argument("--out"); ap.add_argument("--check-against", help="encoder internal dense .pt {level: [g,u,d]}")
    ap.add_argument("--xcheck-ref15", type=int, default=0, help="units per projection to cross-check with ref15_spec")
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(12 / 80)
    art = torch.load(a.artifact, weights_only=False, map_location="cpu")
    W = decode_expert(art, a.level)
    if a.out:
        torch.save({k: w.cpu() for k, w in zip(["gate", "up", "down"], W)}, a.out)
    if a.check_against:
        ref = torch.load(a.check_against, weights_only=False)[a.level]
        print("bit-exact:", all(torch.equal(x.cpu(), y.cpu()) for x, y in zip(W, ref)))
    if a.xcheck_ref15:
        for p in ("gate", "up", "down"):
            print(p, "ref15 mismatches:", xcheck_ref15(art[p], a.xcheck_ref15))


if __name__ == "__main__":
    main()
