"""T35 Stage 1: pattern-rate BASE for the NestQuant v1 encoder/decoder (thread 12), as an import-time extension.

Thread 12's production encoder hard-wires the base at K=2 in four places (base_quant's Viterbi + stream check, prep's
global scale search, pack's base stream, and the decoder's ring_levels / xcheck_ref15).  This module re-implements
those four functions with a base K taken from NE.BASE_K (module global, default 2 = byte-identical to production) and
installs them over nq_encode / nq_decode.  Nothing in threads/12 is edited; production callers that never import
this module are unaffected.  The base K is recorded in planes["meta"]["base_K"] and the decoder reads it from there
(absent -> 2), so artifacts are self-describing.

Base patterns (kernel masks, w_p = KA + bit(p % 16)):  1.5 = (1, 0xAAAA), 1.75 = (1, 0xEEEE), 1.25 = (1, 0x8888).
Matched residuals (total trellis 4.1042 bpw like today's 2 + (2, 2, 2.3125)):
    b1.75: gate/up 2.25 (2, 0x8888), down 2.5625 (2, bres(9))
    b1.5 : gate/up 2.5  (2, 0xAAAA), down 2.8125 (2, bres(13))
bres(n) = Bresenham-spread n-of-16 mask (bres(5) == 0x9248, today's production down pattern).
The fold (ref15 Mb/N int fold on the 16-bit state hashes) is K-independent; only the state walk changes.
"""
import os, sys
T12 = "/home/coder/git/nestquant/threads/12-reference-encoder"
if T12 not in sys.path:
    sys.path.insert(0, T12)
import torch
import nq_decode as D
import nq_encode as NE
import nq_patvit as PV
import harness as h


def bres(n):
    """n-of-16 evenly spread kernel mask: bit j set iff floor((j+1) n / 16) > floor(j n / 16)."""
    return sum(1 << j for j in range(16) if ((j + 1) * n) // 16 > (j * n) // 16)


assert bres(5) == 0x9248 and bres(8) == 0xAAAA
NEW = {1.25: (1, 0x8888), 2.5625: (2, bres(9)), 2.8125: (2, bres(13))}
# LDLQ drift (only scales the g-scale search samples): exllamav3 table 1:1.08 1.5:1.035 2:1.018 2.5:1.009 3:1.004,
# log-linear in K for the new rates
DRIFT = {1.25: 1.056, 1.75: 1.026, 2.5625: 1.0077, 2.8125: 1.0055}
CONFIGS = {
    "b20": dict(base_K=2.0, res_K={"gate": 2.0, "up": 2.0, "down": 2.3125}),
    "b175": dict(base_K=1.75, res_K={"gate": 2.25, "up": 2.25, "down": 2.5625}),
    "b15": dict(base_K=1.5, res_K={"gate": 2.5, "up": 2.5, "down": 2.8125}),
}
NE.BASE_K = 2.0


def install():
    for K, v in NEW.items():
        D.PATTERNS.setdefault(K, v)
    Qm = h._ex()
    for K, d in DRIFT.items():
        Qm.LDLQ_DRIFT.setdefault(K, d)
    PV.ext = ext
    NE.base_quant = base_quant
    NE.prep = prep
    NE.pack = pack
    D.ring_levels = ring_levels
    D.xcheck_ref15 = xcheck_ref15


_EXT = None


def ext():
    """PV.ext with this thread's csrc/nq15_fracvit.cu (thread 12's instances + the new (KA, step-mask) patterns)."""
    global _EXT
    if _EXT is None:
        import exllamav3
        import torch.utils.cpp_extension as CE
        from torch.utils.cpp_extension import load
        bd = "/tmp/nestquant/35-nq15/ext"
        os.makedirs(bd, exist_ok=True)
        os.environ.setdefault("CUDA_HOME", "/home/coder/git/glm52/.venv/lib/python3.12/site-packages/nvidia/cu13")
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")
        if CE.CUDA_HOME is None:
            CE.CUDA_HOME = os.environ["CUDA_HOME"]
        nb = "/home/coder/git/glm52/.venv/bin"
        if nb not in os.environ.get("PATH", ""):
            os.environ["PATH"] = nb + ":" + os.environ.get("PATH", "")
        inc = os.path.join(os.path.dirname(exllamav3.__file__), "exllamav3_ext")
        _EXT = load(name="nq15_fracvit", sources=[os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc", "nq15_fracvit.cu")],
                    extra_include_paths=[inc], build_directory=bd, extra_cuda_cflags=["-O3", "-lineinfo"], verbose=False)
    return _EXT


def _Kb():
    return float(NE.BASE_K)


# ------------------------------------------------------------------------------------------------ encoder
def base_quant(R, tgt, var=None):
    """== NE.base_quant with the base Viterbi at NE.BASE_K."""
    Kb = _Kb()
    rings = R.to_rings(tgt)
    tk = R.to_kring(tgt).double()
    tab = D.variant_table(var, tgt.device) if var else torch.ones(1, device=tgt.device, dtype=torch.float64)
    best = None
    nr = rings.shape[0]
    st_all = NE.viterbi(torch.cat([rings / float(a) for a in tab]), Kb)
    for vi in range(len(tab)):
        a = tab[vi]
        sk = R.kstates(st_all[vi * nr:(vi + 1) * nr])
        if NE.CHECK_STREAMS:
            sk, _ = NE.roundtrip(sk, Kb)
        q = D.q2_values(D.hsum(sk), a if var else None)
        m = (q - tk).square().sum(-1)
        if best is None:
            best, bsk, bq = m, sk, q
            bsel = torch.zeros_like(m, dtype=torch.long)
        else:
            w = m < best
            best = torch.where(w, m, best); bsel = torch.where(w, vi, bsel)
            bsk = torch.where(w.unsqueeze(-1), sk, bsk); bq = torch.where(w.unsqueeze(-1), q, bq)
    av = tab[bsel].unsqueeze(-1) if var else None
    return R.to_unit(bq), bsk, bsel.to(torch.uint8), av


_prep_orig = NE.prep


@torch.no_grad()
def prep(W, H, count, sigma, **kw):
    """NE.prep, then redo the global scale search at the base K (identity when BASE_K == 2)."""
    P = _prep_orig(W, H, count, sigma, **kw)
    Kb = _Kb()
    if Kb == 2.0:
        return P
    Qm = h._ex()
    g0 = P["gs"]
    w0 = P["weight"] / g0                                     # the pre-gs rotated weight (same sampling as prep)
    samp = Qm.sample_scale_tiles(w0, 3)
    q = PV.patq if PV.is_pat(Kb) else h.ExtTileQuantizer("mul1")
    gs, _ = h._g_scale_search(samp * Qm.ldlq_drift(NE._k(Kb)), NE._k(Kb), q)
    P["weight"] = w0 * gs
    P["su"] = P["su"] * g0 / gs
    P["gs"] = gs; P["gs_K2"] = g0; P["base_K"] = Kb
    return P


_pack_orig = NE.pack


@torch.no_grad()
def pack(P, enc, meta, scales):
    """NE.pack, then re-pack the base streams at the base K and record base_K in meta."""
    planes = _pack_orig(P, enc, meta, scales)
    Kb = _Kb()
    if Kb == 2.0:
        return planes
    dev = enc["Q2"].device
    order, shard = D.unit_order(enc["tk"], enc["tn"], meta["shard_axis"], dev)
    sb = enc["sb"].view(-1, 8, 256).long()
    shard_sorted = shard[order]
    planes["base"]["shards"] = [D.pack_stream(D.symbols(sb[order[shard_sorted == s]].view(-1, 256), Kb), Kb).flatten().cpu()
                                for s in range(int(shard.max().item()) + 1)]
    planes["meta"]["base_K"] = Kb
    return planes


# ------------------------------------------------------------------------------------------------ decoder
def base_K_of(P):
    return float(P["meta"].get("base_K", 2))


@torch.no_grad()
def ring_levels(P, device="cuda"):
    """== D.ring_levels with the base streams read at meta base_K."""
    m = P["meta"]; tk, tn = m["tk"], m["tn"]
    order, _ = D.unit_order(tk, tn, m["shard_axis"], device)
    U = tk * tn
    sb = D._gather_streams(D._cat(P["base"]["shards"], device), torch.full((U,), base_K_of(P), device=device), device)
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
    return dict(Sb=Sb, Sr=Sr, a=a, Mb=Mb, N=N, Q2=Q2, Q4=Q4, order=order, Ku=Ku)


def _r15K(K):
    """ref15_spec pattern key: explicit (KA, MASK) (ref15 has its own, smaller table)."""
    return D.PATTERNS[float(K)] if float(K) in D.PATTERNS else int(K)


@torch.no_grad()
def xcheck_ref15(P, nunits=16, seed=0):
    """== D.xcheck_ref15 with the base stream at meta base_K (thread-15 numpy ref15_spec.decode_unit, independent)."""
    sys.path.insert(0, "/home/coder/git/nestquant/threads/15-level4-decode")
    import ref15_spec as R15
    m = P["meta"]
    rl = ring_levels(P, "cpu")
    U = m["tk"] * m["tn"]
    Kb = base_K_of(P)
    base = D._cat(P["base"]["shards"], "cpu"); res = D._cat(P["p4"]["shards"], "cpu")
    Ks = rl["Ku"]
    bper = torch.full((U,), 8 * D.ring_bytes(Kb), dtype=torch.long)
    rper = torch.tensor([8 * D.ring_bytes(float(k)) for k in Ks.tolist()])
    boff = torch.cumsum(bper, 0) - bper; roff = torch.cumsum(rper, 0) - rper
    g = torch.Generator().manual_seed(seed)
    bad = 0
    for u in torch.randint(0, U, (nunits,), generator=g).tolist():
        bs = base[boff[u]:boff[u] + bper[u]].view(8, -1).numpy()
        rs = res[roff[u]:roff[u] + rper[u]].view(8, -1).numpy()
        q2, q4 = R15.decode_unit(bs, rs, int(rl["Mb"][u]), int(rl["N"][u]), Kb=_r15K(Kb), Kr=_r15K(Ks[u]))
        if rl["a"] is not None:
            if m["base_var"] != "sign":
                return None
            a = rl["a"][u].numpy()
            q2, q4 = a * q2, a * q4
        bad += int((q2 != rl["Q2"][u].numpy()).sum() + (q4 != rl["Q4"][u].numpy()).sum())
    return bad


install()
