"""NestQuant quality harness, anchored to EXL3 (exllamav3 1.5.1, as installed in the glm52 venv).

One expert at a time.  Everything is memory bounded: teacher weights are sliced out of the FP8/MXFP4
shards by key, statistics and captures are mmapped, evaluation runs in batches of 64 rows.

Library API (import harness):
    data = load_expert(layer=16, expert=36)                     # teacher + matched statistics + captures
    q = load_exl3_bin(path)                                     # [gate, up, down] dequantized fp32 [out, in]
    q = load_nvfp4(path, data)
    proxy_losses(data, q)                                       # per projection tr(E H E^T)/tr(W H W^T)
    evaluate(data, {"mine": q, ...})                            # forced / routed rel. output L2 per domain
    Wq, info = quantize_exl3_like(W, H, K, count=..., **knobs) # EXL3 reimplementation with ablation knobs
    q = quantize_expert_exl3_like(data, K, **knobs)             # all three projections, same Hessian protocol
    free_scratch()                                              # drop exllamav3's cached Viterbi scratch

CLI (use ./run.sh, which sources the glm52 runtime env; see `./run.sh -h`):
    python harness.py eval   --layer 16 --expert 36 [--exl3-dir D] [--nvfp4 F] [--method name=path.pt ...]
    python harness.py fit    --layer 16 --expert 36 --K 2 [--backend reimpl|upstream] [--codebook mul1] ...
    python harness.py selftest

Conventions: weights are torch float32 [out_features, in_features] (HF layout).  gate/up use
H_in = grams[0] (6144x6144), down uses grams[1] (2048x2048); grams are sum_rows (p x)(p x)^T with p the
router weight, i.e. router-weight-squared weighting, divided by training_rows before use.
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
import sys, json, math, time, argparse, hashlib, resource
from pathlib import Path

ORBIT = "/home/coder/git/orbit-duet"
if ORBIT not in sys.path:
    sys.path.insert(0, ORBIT)

import torch
import torch.nn.functional as F

# The installed exllamav3 wheel is cu128 but torch is cu130: preload a CUDA 12 runtime (kept next to this
# file) so exllamav3_ext resolves libcudart.so.12.  Harmless if one is already on the loader path.
_CU12 = Path(__file__).resolve().parent / "lib" / "libcudart.so.12"
if _CU12.exists():
    import ctypes
    ctypes.CDLL(str(_CU12), mode=ctypes.RTLD_GLOBAL)

_GPU_CAPPED = False


def gpu_cap(gib=12):
    """Brief rule: at most 12 GB on the assigned GPU."""
    global _GPU_CAPPED
    if not _GPU_CAPPED and torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        torch.cuda.set_per_process_memory_fraction(min(1.0, gib / total))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_num_threads(8)
        _GPU_CAPPED = True


# --------------------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------------------
GLM_SOURCE = os.environ.get("NQ_GLM_SOURCE", "/tmp/nestquant/glm53-fp8-experts")  # per-expert FP8 files (box restart 2026-09-28 wiped the full copy)
GLM_RUN = ORBIT + "/runs/glm53_pilot_matched_l{L}"
GLM_CAPTURES = {
    # The ONLY GLM evaluation capture (sha df47ef..., layers 16/32/49/66, 5120 rows).  It carries both
    # control:* and ood:* domains; the "control"/"ood" groups in evaluate() are domain-prefix subsets of it.
    # NB: runs/native_id_control_v1_capture and runs/ood_controlled_v1_capture are MiMo captures
    # (384 experts) -- do not use them for GLM.
    "matched": ORBIT + "/runs/glm53_matched_context_pilot_v1_capture/layer_{L}.pt",
}
NUMEL = 3 * 2048 * 6144


class ExpertData:
    """Teacher weights, matched training statistics and frozen evaluation captures for one expert."""

    def __init__(self, layer, expert, teacher, stats, capture=None, capture_path=None, stats_path=None):
        self.layer, self.expert = layer, expert
        self.teacher = teacher                         # [g, u, d] fp32 cuda [out, in]
        self.stats = stats                             # orbit_duet load_statistics dict (mmapped)
        self.count = stats["metadata"]["training_rows"]
        self.capture = capture
        self.capture_path, self.stats_path = capture_path, stats_path

    def H(self, proj, normalized=True):
        """Input Gram for projection 0/1/2 (gate/up/down); normalized = divided by training_rows."""
        g = self.stats["grams"][0 if proj < 2 else 1]
        g = g.to("cuda", torch.float32)
        return g / self.count if normalized else g

    @property
    def identity(self):
        return self.stats["identity"]


def load_expert(layer, expert, source=GLM_SOURCE, statistics=None, capture="matched"):
    """Load one expert.  `capture` is a key of GLM_CAPTURES, a path, or None (no output eval)."""
    gpu_cap()
    from orbit_duet.source import weights
    from orbit_duet.statistics import load_statistics
    teacher = weights(source, layer, expert)
    statistics = statistics or f"{GLM_RUN.format(L=layer)}/statistics/l{layer}_e{expert}.pt"
    stats = load_statistics(statistics, teacher, layer, expert)       # verifies teacher hashes
    cap = cap_path = None
    if capture is not None:
        cap_path = GLM_CAPTURES.get(capture, capture).format(L=layer)
        cap = torch.load(cap_path, weights_only=True, mmap=True)
        if cap["layer"] != layer or cap["protocol"]["role"] != "evaluation only":
            raise ValueError("frozen evaluation-only capture of the same layer required")
    return ExpertData(layer, expert, teacher, stats, cap, cap_path, statistics)


def load_exl3_bin(path):
    """Dequantize an orbit-duet EXL3 .bin (4096-byte JSON header) with exllamav3's own decoder."""
    gpu_cap()
    from orbit_duet.exl3_adapter import EXL3Expert
    return [w.float().contiguous() for w in EXL3Expert(str(path)).decoded_weights()]


def load_nvfp4(path, data):
    from orbit_duet.modelopt_nvfp4 import load_artifact
    q, _ = load_artifact(path, data.teacher, data.layer, data.expert, data.identity)
    return [w.float() for w in q]


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------------------
@torch.no_grad()
def proxy_loss(W, Wq, H):
    """tr(E H E^T) / tr(W H W^T) with W, Wq [out, in] and H [in, in]."""
    W = W.cuda().float(); E = Wq.cuda().float() - W
    num = (E @ H * E).sum(dtype=torch.float64).item()
    den = (W @ H * W).sum(dtype=torch.float64).item()
    return num / den


@torch.no_grad()
def proxy_losses(data, q, sigma_reg=0.0):
    """Per-projection proxy loss.  sigma_reg>0 adds EXL3's damping sigma*mean(diag H) (default: raw H)."""
    out = {}
    Hs = {}
    for i, name in enumerate(["gate", "up", "down"]):
        key = 0 if i < 2 else 1
        if key not in Hs:
            H = data.H(i)
            if sigma_reg:
                H.diagonal().add_(sigma_reg * H.diagonal().mean())
            Hs[key] = H
        out[name] = proxy_loss(data.teacher[i], q[i], Hs[key])
    return out


def _teacher(x, w):
    # identical to orbit_duet.evaluate.teacher: BF16 expert math, fp32 result
    g, u, d = w
    return F.linear(F.silu(F.linear(x.bfloat16(), g.bfloat16())) * F.linear(x.bfloat16(), u.bfloat16()),
                    d.bfloat16()).float()


@torch.no_grad()
def _errors(capture, teacher, methods, rows, prob, batch=64):
    """Replicates benchmarks.matched_native_expert.errors (router-weighted and unweighted rel. L2)."""
    if not len(rows):
        return None
    num = {k: 0. for k in methods}; plain = {k: 0. for k in methods}; den = plain_den = 0.
    bf = {k: [t.bfloat16() for t in w] for k, w in methods.items()}   # cast once (same values)
    tb = [t.bfloat16() for t in teacher]
    for first in range(0, len(rows), batch):
        ids = rows[first:first + batch]
        x = capture["x"][ids].cuda()
        p2 = prob[first:first + batch].cuda().double().square()
        target = _teacher(x, tb).double()
        energy = target.square().sum(-1)
        den += float((energy * p2).sum()); plain_den += float(energy.sum())
        for k, w in bf.items():
            e = (_teacher(x, w).double() - target).square().sum(-1)
            num[k] += float((e * p2).sum()); plain[k] += float(e.sum())
    return dict(rows=len(rows),
                router_weighted_relative_l2={k: (v / den) ** .5 if den else None for k, v in num.items()},
                unweighted_relative_l2={k: (v / plain_den) ** .5 if plain_den else None for k, v in plain.items()})


@torch.no_grad()
def evaluate(data, methods, groups=True):
    """Relative expert-output L2 on the frozen capture.

    methods: {name: [gate, up, down]} dequantized weights ([out, in]).
    Returns {domain: {"forced": ..., "routed": ...}} where forced = every capture token pushed through the
    expert (weight 1), routed = only tokens whose top-k contains the expert, weighted by router prob.
    Domains: "all", each capture domain, and (groups=True) the aggregates "control" / "ood".
    """
    cap = data.capture
    routed, slots = torch.where(cap["ids"] == data.expert)
    domains = [("all", torch.arange(len(cap["x"])))]
    names = sorted(set(cap["domains"]))
    doc_domain = cap["domains"]
    if groups:
        for g in sorted({n.split(":")[0] for n in names if ":" in n}):
            docs = torch.tensor([i for i, d in enumerate(doc_domain) if d.split(":")[0] == g])
            domains.append((g, torch.isin(cap["document_ids"], docs).nonzero().flatten()))
    for dom in names:
        docs = torch.tensor([i for i, d in enumerate(doc_domain) if d == dom])
        domains.append((dom, torch.isin(cap["document_ids"], docs).nonzero().flatten()))
    out = {}
    for dom, rows in domains:
        take = torch.isin(routed, rows); actual = routed[take]; prob = cap["p"][actual, slots[take]]
        out[dom] = dict(forced=_errors(cap, data.teacher, methods, rows, torch.ones(len(rows))),
                        routed=_errors(cap, data.teacher, methods, actual, prob))
    return out


def table(ev, domains=("all", "control", "ood"), key="router_weighted_relative_l2"):
    """Compact percent table {method: {domain/forced|routed: pct}}."""
    res = {}
    for dom in domains:
        if dom not in ev:
            continue
        for kind in ["forced", "routed"]:
            r = ev[dom][kind]
            if r is None:
                continue
            for m, v in r[key].items():
                res.setdefault(m, {})[f"{dom}/{kind}"] = None if v is None else round(100 * v, 3)
    return res


# --------------------------------------------------------------------------------------------------
# EXL3 reimplementation with knobs
# --------------------------------------------------------------------------------------------------
def _ex():
    from exllamav3.modules.quant.exl3_lib import quantize as Q
    return Q


def free_scratch():
    """Drop exllamav3's lru-cached Viterbi scratch (get_temp_buffers is cached per (device, K, codebook):
    ~2 GB at K=2, 1 GB at K=3, 0.5 GB at K=4).  Call between codebook/K sweeps to stay under the VRAM cap."""
    Qm = _ex()
    for f in ("get_temp_buffers", "get_temp_buffers_frac"):
        if hasattr(getattr(Qm, f, None), "cache_clear"):
            getattr(Qm, f).cache_clear()
    torch.cuda.empty_cache()


def codebook_lut(codebook="mul1", device="cuda"):
    """Value of every 16-bit trellis state, bit-exact with exllamav3's CUDA decoders (fp16 arithmetic).

    '3inst' (cb 0, QTIP 3INST): x = s*89226354 + 64248484;  x = (x & 0x8fff8fff) ^ 0x3b603b60;  hi+lo as fp16
    'mcg'   (cb 1):             x = s*0xCBAC1FED;          same lop3 + fp16 add
    'mul1'  (cb 2, default):    x = s*0x83DCD12D;          v = fp16(1024 + bytesum(x)) * (1/147.7) - 10.39 (hfma)
    """
    s = torch.arange(65536, dtype=torch.int64)
    M = 2**32
    if codebook in ("3inst", "mcg"):
        x = (s * 89226354 + 64248484) % M if codebook == "3inst" else (s * 0xCBAC1FED) % M
        x = (x & 0x8fff8fff) ^ 0x3b603b60
        lo_bits = (x & 0xFFFF).to(torch.int32).to(torch.int16)
        hi_bits = (x >> 16).to(torch.int32).to(torch.int16)
        lo = lo_bits.view(torch.float16).float(); hi = hi_bits.view(torch.float16).float()
        v = (lo + hi).half()                       # __hadd: exact sum rounded once to fp16
    elif codebook == "mul1":
        x = (s * 0x83DCD12D) % M
        bsum = (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + ((x >> 24) & 255)
        h = (1024 + bsum).double()                  # exact in fp16 (1024..2044)
        k_inv = torch.tensor([0x1eee], dtype=torch.int16).view(torch.float16).double()
        k_bias = torch.tensor([0xc931 - 65536], dtype=torch.int16).view(torch.float16).double()
        v = (h * k_inv + k_bias).half()             # hfma: single rounding
    else:
        raise ValueError(codebook)
    return v.float().to(device)


class ExtTileQuantizer:
    """exllamav3's CUDA Viterbi (fp16 costs, tail-biting via two rolled passes). codebook in 3inst/mcg/mul1."""

    def __init__(self, codebook="mul1"):
        self.codebook = codebook

    def __call__(self, tiles, K):
        qa = {"K": K}
        if self.codebook == "mul1": qa["mul1"] = True
        elif self.codebook == "mcg": qa["mcg"] = True
        return _ex().quantize_tiles(tiles, qa)


class TorchTileQuantizer:
    """Reference Viterbi in PyTorch for an arbitrary 65536-entry codebook LUT (fp32 costs).

    Same trellis as EXL3: 16-bit shift-register state, K new bits shifted in at the bottom per weight
    (state_t = ((state_{t-1} << K) | b_t) & 0xFFFF), value = lut[state_t], 256-step tail-biting ring solved
    with EXL3's two-pass trick (pass 1 rolled by 128, trace back the state entering position 0, pass 2
    constrained to start from and end in it).  Slow (~10x the CUDA kernel) but codebook/K agnostic.
    """

    def __init__(self, lut, max_bytes=1 << 31):
        self.lut = lut.float().cuda()
        self.max_bytes = max_bytes

    def __call__(self, tiles, K):
        T, L = tiles.shape
        E = 65536 >> K
        chunk = max(1, min(T, self.max_bytes // (L * E)))
        qs, ids = [], []
        for a in range(0, T, chunk):
            q, i = self._run(tiles[a:a + chunk].float(), K)
            qs.append(q); ids.append(i)
        return torch.cat(qs), torch.cat(ids)

    def _run(self, w, K):
        T, L = w.shape
        dev = w.device
        Kr = 16 - K; E = 1 << Kr; Q = 1 << K
        e = torch.arange(E, device=dev)
        k = torch.arange(Q, device=dev)
        states = (k[:, None] << Kr) | e[None, :]              # [Q, E] full 16-bit state for (top bits k, edge e)
        prev = states >> K                                     # [Q, E] predecessor edge
        vals = self.lut[states]                                # [Q, E]
        back = torch.empty((L, T, E), dtype=torch.uint8, device=dev)
        inf = torch.tensor(float("inf"), device=dev)

        def forward(roll, pre):
            cost = None
            for i in range(L):
                ri = (i + roll) % L
                d = (vals[None] - w[:, ri, None, None]).square()          # [T, Q, E]
                if cost is None:
                    if pre is not None:
                        d = torch.where(prev[None] == pre[:, None, None], d, inf)
                    c = d
                else:
                    c = d + cost[:, prev]                                  # gather [T, Q, E]
                cost, arg = c.min(1)
                back[ri] = arg.to(torch.uint8)
            return cost

        def trace(roll, edge, stop_at_zero):
            out = torch.empty((T, L), dtype=torch.int64, device=dev)
            tt = torch.arange(T, device=dev)
            for i in range(L - 1, -1, -1):
                ri = (i + roll) % L
                kk = back[ri][tt, edge].long()
                st = (kk << Kr) | edge
                out[:, ri] = st
                edge = st >> K
                if stop_at_zero and ri == 0:
                    break
            return out, edge

        c = forward(L // 2, None)
        _, start = trace(L // 2, c.argmin(1), True)            # edge entering position 0
        c = forward(0, start)
        idx, _ = trace(0, start, False)                         # ring closes on `start`
        return self.lut[idx], idx.to(torch.int16)


def _had(n):
    return _ex().get_hadamard_dt(n, "cuda", torch.float, 1 / math.sqrt(n))


def _g_scale_search(samples, K, quantizer):
    """exllamav3 g_scale_search_batch (coarse 0.1..1.9 grid on 1/3 of tiles, fine +-2x0.075, parabolic)."""
    def mse(tiles, s):
        q, _ = quantizer(tiles * s, K)
        return (q / s - tiles).square().mean().item()
    coarse = [0.1 + 0.2 * i for i in range(10)]
    sub = samples[::3]
    c = coarse[min(range(10), key=lambda i: mse(sub, coarse[i]))]
    step = 0.075
    fine = [c + step * (i - 2) for i in range(5)]
    m = [mse(samples, s) for s in fine]
    best = min(range(5), key=lambda i: m[i])
    off = 0.0
    if 0 < best < 4:
        den = m[best - 1] - 2 * m[best] + m[best + 1]
        off = max(-.5, min(.5, 0.5 * (m[best - 1] - m[best + 1]) / den)) if den > 0 else 0.0
    return max(fine[best] + off * step, 0.01), m[best]


@torch.no_grad()
def quantize_exl3_like(W, H, K, count=1, *, seed=91426, sigma_reg=0.03, codebook="mul1", quantizer=None,
                       apply_out_scales=None, g_scale=True, g_scale_K=None, refit=True, ldlq=True,
                       buf_size_k=128, fp16_scales=True, return_info=True):
    """Faithful EXL3 quantization of one linear, reimplemented around a pluggable tile quantizer.

    W:  [out, in] teacher weight (any float dtype/device).     H: [in, in] input Gram (sum x x^T).
    count: rows in H (H/count is used; irrelevant for quality since damping is relative).
    K:  bits per weight (int 1..8, half-integer with mul1), OR a sequence with one K per 16-row block of the
        input dimension (length in/16; LDLQ block j covers input channels 16j..16j+15 in the *rotated* basis).
    Knobs (defaults = the orbit-duet matched EXL3 protocol, which reproduces the .bin files bit-exactly):
        seed=91426        torch.manual_seed before the su / sv sign draws
        sigma_reg=0.03    H += sigma * mean(diag H) * I
        codebook          'mul1' | 'mcg' | '3inst' | a 65536-float LUT tensor (-> TorchTileQuantizer)
        quantizer         callable(tiles[T,256], K) -> (q_tiles, idx) overriding `codebook`
        apply_out_scales  None = EXL3 skew rule (<15% of sqrt-diag mass on 2% channels), or True/False
        g_scale           run the global scale search (False -> 1.0);  g_scale_K: K used for it (per-block K)
        refit             closed-form post-LDLQ su/sv refit in the Hessian metric (exllamav3 >= 1.5)
        ldlq              False -> no error feedback (plain per-tile trellis rounding)
        fp16_scales       store suh/svh as fp16 and decode like exllamav3 (fp16 had passes)
    Returns (Wq [out, in] fp32, info) where info has proxy (upstream rotated/damped pre-refit),
    tensors (suh, svh, indices [in/16, out/16, 256] int16, trellis if packable), g_scale, apply_out_scales,
    bits (trellis + fp16 scale bits).
    """
    gpu_cap()
    Qm = _ex()
    dev = torch.device("cuda")
    if quantizer is None:
        quantizer = TorchTileQuantizer(codebook) if torch.is_tensor(codebook) else ExtTileQuantizer(codebook)
    weight = W.to(dev, torch.float32).T.contiguous()                  # EXL3 layout (k=in, n=out)
    k, n = weight.shape
    Ks = [K] * (k // 16) if not isinstance(K, (list, tuple)) and not torch.is_tensor(K) else [
        (int(x) if float(x).is_integer() else float(x)) for x in K]
    assert len(Ks) == k // 16
    torch.manual_seed(seed)

    # ---- Hessian: mean, damping, input signs, 128-block Hadamard, block-16 LDL
    Hm = H.to(dev, torch.float32).clone() / count
    diag_mean = torch.diag(Hm).mean().item()  # torch.diag (not .diagonal()) = upstream reduction order, bit-exact
    Hm.diagonal().add_(sigma_reg * diag_mean)
    H_diag = Hm.diagonal().clone()
    su = (torch.randn(k, device=dev).sign() + 1e-5).sign().float().unsqueeze(1)
    su_signs = su.clone()
    Hm *= su.T; Qm.blockwise_preapply_had_r_(Hm, 128); Hm *= su; Qm.blockwise_preapply_had_l_(Hm, 128)
    Lf, Hr = Qm.block_ldl(Hm, 16, {"sigma_reg": sigma_reg}, False)
    Lf.diagonal().zero_()
    Hr = Hr.to(dev)
    sv = (torch.randn(n, device=dev).sign() + 1e-5).sign().float().unsqueeze(0)
    weight_orig = weight.clone()

    # ---- regularize (exllamav3.regularize with pluggable scale search)
    d = torch.sort(H_diag.sqrt(), descending=True).values
    skew = (d[:k // 50].sum() / d.sum()).item()
    aos = (skew < 0.15) if apply_out_scales is None else apply_out_scales
    ocs = Qm.block_rms(weight, dim=0, keepdim=True)
    ocs /= ocs.mean().item()
    zero = ocs.abs() < 1e-30
    if aos:
        ocs[zero] = 0.1
        sv = (sv * ocs + 1e-10).float()
    weight /= sv
    sv[zero] = 0.0
    Qm.blockwise_preapply_had_r_(weight, 128)
    ics = Qm.block_rms(weight, dim=1, keepdim=True)
    ics[ics.abs() < 1e-30] = 0.1
    su = (su * ics / (-Qm.codebook_scale) + 1e-10).float()
    weight /= su
    Qm.blockwise_preapply_had_l_(weight, 128)
    gs = 1.0
    if g_scale:
        gK = g_scale_K if g_scale_K is not None else (Ks[0] if len(set(Ks)) == 1 else max(set(Ks), key=Ks.count))
        tiles = Qm.sample_scale_tiles(weight, 3) * Qm.ldlq_drift(gK)
        gs, _ = _g_scale_search(tiles, gK, quantizer)
    weight *= gs
    su /= gs

    # ---- LDLQ over 16-row blocks, last block first (exllamav3.ldlq with per-block K)
    perm = Qm.tensor_core_perm(dev); perm_i = Qm.tensor_core_perm_i(dev)
    tiles_n = n // 16
    wq = torch.zeros_like(weight)
    enc = torch.zeros((k // 16, tiles_n, 256), dtype=torch.int16, device=dev)
    prod = torch.zeros_like(weight)
    Lz = Lf if ldlq else torch.zeros_like(Lf)
    torch.cuda.synchronize()
    stream = Qm.get_quant_stream(dev)       # same stream as exllamav3 (cuBLAS results are stream-sensitive)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
      for j in range(k, 0, -buf_size_k):
        i = j - buf_size_k
        bw = weight[i:j]; bq = wq[i:j]; bL = Lz[i:j]
        for bj in range(buf_size_k, 0, -16):
            bi = bj - 16
            comp = prod[i + bi:i + bj]
            comp.addmm_(bL[bj:, i + bi:i + bj].T, bw[bj:] - bq[bj:])
            rows = bw[bi:bj] + comp
            t = rows.reshape(16, tiles_n, 16).permute(1, 0, 2).reshape(tiles_n, 256)[:, perm]
            qt, qi = quantizer(t.contiguous(), Ks[(i + bi) // 16])
            bq[bi:bj] = qt[:, perm_i].reshape(tiles_n, 16, 16).permute(1, 0, 2).reshape(16, n)
            enc[(i + bi) // 16] = qi
        prod.addmm_(bL.T, bw - bq)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    del prod, Lz, Lf
    E = weight - wq
    proxy = Qm.block_trace(E, Hr) / max(Qm.block_trace(weight, Hr), 1e-8)
    del E

    # ---- back to the original basis, scale refit
    Wr = wq.clone()
    Wr = Qm.preapply_had_l(Wr, 128); Wr *= su; Wr = Qm.preapply_had_r(Wr, 128); Wr *= sv
    if refit:
        H_orig = Qm.unrotate_H(Hr.cpu(), su_signs.cpu())  # upstream unrotates on CPU (H parked there); bit-exact
        Wr, su, sv, _, _ = Qm.refit_scales(weight_orig, Wr, H_orig, su, sv)
        su = su.view(-1, 1); sv = sv.view(1, -1)
        del H_orig
    suh = su.flatten().half(); svh = sv.flatten().half()
    if fp16_scales:     # decode exactly like LinearEXL3.get_weight_tensor
        w = wq.half()
        w = Qm.preapply_had_l(w, 128); w *= suh.unsqueeze(1)
        w = Qm.preapply_had_r(w, 128); w *= svh.unsqueeze(0)
        Wq = w.float().T.contiguous()
    else:
        Wq = Wr.T.contiguous()
    info = None
    if return_info:
        tensors = dict(suh=suh, svh=svh, indices=enc)
        if len(set(Ks)) == 1 and not torch.is_tensor(codebook):
            tensors["trellis"] = Qm.pack_trellis(enc, {"K": Ks[0]})
            if codebook in ("mul1", "mcg"):
                mult = Qm.codebook_mul1_mult if codebook == "mul1" else Qm.codebook_mcg_mult
                tensors[codebook] = torch.tensor(mult, dtype=torch.uint32).view(torch.int)
        bits = 256 * sum(Ks) * tiles_n + 16 * (k + n)
        info = dict(proxy=proxy, g_scale=gs, apply_out_scales=bool(aos), skew=skew, tensors=tensors,
                    bits=bits, bpw=bits / (k * n), K=Ks)
    return Wq, info


@torch.no_grad()
def quantize_exl3_upstream(W, H, K, count=1, seed=91426, sigma_reg=0.03, codebook="mul1"):
    """Call exllamav3's own quantize_exl3 exactly as orbit-duet's fit_matched_exl3 does."""
    gpu_cap()
    Qm = _ex()
    qa = dict(K=K, devices=["cuda:0"], seed=seed, sigma_reg=sigma_reg, apply_out_scales=None)
    qa[codebook] = True
    hdata = dict(H=H.to("cuda", torch.float32).clone(), count=count, finalized=False, device=torch.device("cuda:0"))
    _, proxy, tensors = Qm.quantize_exl3(W.to("cuda", torch.float32).T.contiguous(), hdata, qa, False, verbose=False)
    Qm.get_temp_buffers.cache_clear()
    from exllamav3.modules.quant.exl3 import LinearEXL3
    lin = LinearEXL3(None, W.shape[1], W.shape[0], **{k: v for k, v in tensors.items()})
    Wq = lin.get_weight_tensor().T.float().contiguous()
    return Wq, dict(proxy=proxy, tensors=tensors, g_scale=qa["g_scale"], apply_out_scales=qa["apply_out_scales"])


def quantize_expert_exl3_like(data, K, backend="reimpl", **knobs):
    """Quantize gate/up/down of `data` with the matched EXL3 protocol. K may be a scalar or
    {"gate": K, "up": K, "down": K} (each a scalar or per-block list). Returns ([g, u, d], infos)."""
    out, infos = [], []
    for i, name in enumerate(["gate", "up", "down"]):
        Ki = K[name] if isinstance(K, dict) else K
        H = data.H(i, normalized=False)
        fn = quantize_exl3_like if backend == "reimpl" else quantize_exl3_upstream
        Wq, info = fn(data.teacher[i], H, Ki, count=data.count, **knobs)
        del H; torch.cuda.empty_cache()
        out.append(Wq); infos.append(info)
    return out, infos


def write_exl3_bin(path, infos, shapes=((2048, 6144), (2048, 6144), (6144, 2048))):
    """Serialize like orbit-duet fit_matched_exl3 (loadable by load_exl3_bin)."""
    from orbit_duet.exl3_adapter import write_legacy
    vals = [dict(shape=list(s), **{k: v.cpu() for k, v in info["tensors"].items() if k != "indices"})
            for s, info in zip(shapes, infos)]
    return write_legacy(path, vals)


# --------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------
def _load_method(spec, data):
    name, path = spec.split("=", 1)
    if path.endswith(".bin"):
        return name, load_exl3_bin(path)
    obj = torch.load(path, weights_only=True, map_location="cuda")
    if isinstance(obj, dict):
        obj = [obj[k] for k in ["gate", "up", "down"]]
    return name, [t.float().cuda() for t in obj]


def cmd_eval(a):
    t0 = time.time()
    data = load_expert(a.layer, a.expert, a.source, a.statistics, a.capture)
    methods = {}
    run = Path(GLM_RUN.format(L=a.layer))
    exl3_dir = Path(a.exl3_dir) if a.exl3_dir else run / f"exl3_e{a.expert}"
    nv = a.nvfp4 or str(run / f"nvfp4_e{a.expert}/weights.pt")
    if not a.no_refs:
        for bits in [2, 4]:
            p = exl3_dir / f"expert_{bits}.bin"
            if p.exists():
                methods[f"exl3_{bits}"] = load_exl3_bin(p)
        if Path(nv).exists():
            methods["nvfp4"] = load_nvfp4(nv, data)
    for spec in a.method or []:
        n, q = _load_method(spec, data); methods[n] = q
    report = dict(layer=a.layer, expert=a.expert, statistics=data.identity, capture=data.capture_path,
                  proxy={m: proxy_losses(data, q) for m, q in methods.items()})
    ev = evaluate(data, methods)
    report["table_pct"] = table(ev)
    report["evaluation"] = ev
    report["seconds"] = time.time() - t0
    report["peak_cuda_mib"] = torch.cuda.max_memory_allocated() / 2**20
    report["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(json.dumps(dict(proxy=report["proxy"], table_pct=report["table_pct"]), indent=1))
    if a.output:
        Path(a.output).write_text(json.dumps(report, indent=1))


def cmd_fit(a):
    t0 = time.time()
    data = load_expert(a.layer, a.expert, a.source, a.statistics, a.capture)
    knobs = dict(seed=a.seed, sigma_reg=a.sigma_reg, codebook=a.codebook)
    if a.backend == "reimpl":
        knobs.update(refit=not a.no_refit, ldlq=not a.no_ldlq, g_scale=not a.no_g_scale)
    K = float(a.K) if "." in a.K else int(a.K)
    q, infos = quantize_expert_exl3_like(data, K, backend=a.backend, **knobs)
    methods = {f"fit_K{a.K}": q}
    if a.compare_bin:
        methods["reference"] = load_exl3_bin(a.compare_bin)
        print("max |fit - reference| per projection:",
              [float((x - y).abs().max()) for x, y in zip(q, methods["reference"])])
    rep = dict(proxy={m: proxy_losses(data, w) for m, w in methods.items()},
               upstream_proxy=[i["proxy"] for i in infos], g_scale=[i["g_scale"] for i in infos])
    if data.capture is not None:
        rep["table_pct"] = table(evaluate(data, methods))
    rep["seconds"] = time.time() - t0
    print(json.dumps(rep, indent=1))
    if a.save_bin:
        write_exl3_bin(a.save_bin, infos)
    if a.save_pt:
        torch.save({k: v.cpu() for k, v in zip(["gate", "up", "down"], q)}, a.save_pt)
    if a.output:
        Path(a.output).write_text(json.dumps(rep, indent=1))


def cmd_selftest(a):
    gpu_cap()
    torch.manual_seed(0)
    # 1. codebook LUTs match the CUDA kernel's reconstructed values at its chosen indices
    tiles = torch.randn(64, 256, device="cuda")
    for cb in ["mul1", "mcg", "3inst"]:
        lut = codebook_lut(cb)
        for K in [2, 4]:
            q, idx = ExtTileQuantizer(cb)(tiles, K)
            ok = torch.equal(lut[idx.long() & 0xFFFF], q)
            print(f"lut {cb} K={K}: exact={ok}  kernel mse={(q - tiles).square().mean():.5f}")
    # 2. torch Viterbi vs CUDA kernel (same codebook): mse should agree (fp32 vs fp16 costs)
    for K in [2, 3]:
        qk, _ = ExtTileQuantizer("mul1")(tiles[:16], K)
        qt, it = TorchTileQuantizer(codebook_lut("mul1"))(tiles[:16], K)
        st = it.long() & 0xFFFF
        ring = bool(((st.roll(1, 1) & ((1 << (16 - K)) - 1)) == (st >> K)).all())
        print(f"torch viterbi K={K}: mse {(qt - tiles[:16]).square().mean():.5f} vs kernel "
              f"{(qk - tiles[:16]).square().mean():.5f}; tail-biting ring valid={ring}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ["eval", "fit"]:
        s = sub.add_parser(name)
        s.add_argument("--layer", type=int, default=16); s.add_argument("--expert", type=int, default=36)
        s.add_argument("--source", default=GLM_SOURCE); s.add_argument("--statistics")
        s.add_argument("--capture", default="matched", help="matched|<path> (GLM: only matched exists)")
        s.add_argument("--output")
    e = sub.choices["eval"]
    e.add_argument("--exl3-dir"); e.add_argument("--nvfp4"); e.add_argument("--no-refs", action="store_true")
    e.add_argument("--method", action="append", help="name=path (.bin EXL3, or .pt list/dict gate,up,down [out,in])")
    f = sub.choices["fit"]
    f.add_argument("--K", default="2"); f.add_argument("--backend", default="reimpl", choices=["reimpl", "upstream"])
    f.add_argument("--codebook", default="mul1"); f.add_argument("--seed", type=int, default=91426)
    f.add_argument("--sigma-reg", type=float, default=0.03)
    f.add_argument("--no-refit", action="store_true"); f.add_argument("--no-ldlq", action="store_true")
    f.add_argument("--no-g-scale", action="store_true")
    f.add_argument("--compare-bin"); f.add_argument("--save-bin"); f.add_argument("--save-pt")
    sub.add_parser("selftest")
    a = p.parse_args()
    dict(eval=cmd_eval, fit=cmd_fit, selftest=cmd_selftest)[a.cmd](a)


if __name__ == "__main__":
    main()
