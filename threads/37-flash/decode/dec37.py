#!/usr/bin/env python3
"""T37 dec37: batched on-policy decoder for zai-org/GLM-5.3-Flash (FP8 checkpoint, optional NVFP4) on 8x A100.

Stock vLLM 0.31 cannot run Flash DSA/KDA on sm_80 (decode/README.md), so this is a self-contained HF-math decoder:
the per-layer modules ARE transformers.models.glm5_next (as capture37g.py builds them); only the caches, the batching
and the routed-expert execution are ours.

  * single process, layer pipeline over the devices (contiguous layer split balanced on weights + per-slot state);
    one batch of NS slots traverses all stages per step.
  * --ep (expert parallel, default off until GPU-verified): instead of micro-batch pipelining (dequant is ~90% of a
    step and each micro-batch still hits most experts, so G groups ~ G x the dequant work) every MoE layer's routed
    experts are split into fixed blocks sharded over ALL devices; the layer's tokens / weights / expert offsets are
    copied to every shard device, each expert is dequantised exactly once per step, all GPUs work concurrently
    (async launches from one thread, one host sync per layer), and per-assignment fp32 rows come back to the layer
    device to be summed over K in a fixed order -> bitwise identical to non-ep for the same --expert-block.
  * weights: backbone / dense MLP / shared experts / embed / lm_head dequantised once to bf16 and resident; routed
    experts stay FP8 (or NVFP4) on the GPU and are dequantised per block of G experts into a bf16 scratch
    (--dq torch: 2-step fp8->bf16 copy + in-place scale mul, bitwise = capture37g Experts._dq; --dq triton/auto:
    fused one-pass Triton kernel, ~3 B/weight of HBM traffic instead of ~7, bitwise equal (integer RNE);
    both self-checked against the per-expert reference at start, triton falls back to torch on mismatch).
  * MoE: HF router (fp32 sigmoid + correction bias, top-8, norm, x2.5); tokens grouped by expert, padded bmm per
    block (decode) or per-expert linears (prefill); Flash clamps (gate <= 10, |up| <= 10); fp32 accumulation; + shared.
  * KDA (34 layers): own conv state [NS,3,C] + recurrent state [NS,H,dk,dv] fp32; decode = exact fp32 recurrent
    step (HF recurrent_kimi_delta_attention math, TF32 off); prefill = HF chunk_kimi_delta_attention in 64-multiple
    pieces chaining the state (bitwise the same chunk partition as one full call), pads with beta = g = q = k = v = 0.
  * DSA/MLA (11 layers): own absorbed-MLA cache: latent C [NS,Smax,512] (NoPE: qk_rope_head_dim = 0) + indexer
    pooled keys PK [NS,Smax/4,128] (each kpool-4 pool computed once, HF op order) + a 4-token ring for the open pool.
    scores = (q W_uk) . C, out = (p . C) W_uv.  Mask = dense causal while visible pools <= 512 (len <= 2051), else
    indexer top-512 pools + always-selected tail (HF Glm5NextTextIndexer semantics).
  * continuous batching: NS fixed slots, free slots refilled by prefill waves; free slots run a dummy token (cost is
    dominated by the B-independent expert dequant, so a static batch is ~free and avoids gather/scatter of state).
  * routing records for EVERY fed token (prompt + generated) at every MoE layer: ids [n,8] uint16, w [n,8] fp16 (incl
    the 2.5 scaling), xn = sum x^2 of the MoE input (fp32), = the capture37g trace schema.

Output (PRIVATE, never uploaded), OUT = /tmp/nestquant/37-flash/private/dec_trace by default:
  OUT/shards/s{K:05d}/ L{L}.npz (ids, w, xn), tok.npy (int32 per row), aux.npz (ent, lp per row), seqs.json, gen.jsonl
  OUT/ L{L}.r{K}of{W}.npz, tok.r{K}of{W}.npy, seqs.r{K}of{W}.json  (symlinks = the jF decode-trace layout that
      jf/blocks37.py reads; seqs.json = {"N", "seqs": [{"id", "rows", "prompt_len", "group", ...}], "decode": {...}})
  OUT/gen.jsonl (id, group, prompt_len, gen ids, text, stopped, think_end, ...), OUT/index.json (dec.py-style summary)
Rows of a sequence = prompt positions 0..P-1 + generated positions P..P+G-2 (the G-th sampled token, i.e. the stop
token or the max_new-th token, is never fed); tok[row] = the token fed at that row, prompt_len = P.

  dec37.py --input prompts.jsonl --slots 256 --max-new 8192            # gen (see run_dec37.sh)
  dec37.py --smoke --slots 8 --max-new 64 --out /tmp/.../smoke          # built-in prompts
  dec37.py --mode tf --tasks tasks.json --out ...                       # teacher-forced logits (tf_check37.py)
  dec37.py --finalize OUT                                               # (re)build the jF symlink layout
"""
import argparse
import collections
import contextlib
import glob
import json
import math
import os
import re
import signal
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

torch.set_grad_enabled(False)

PFX = "model.language_model."
CK_DEFAULT = "/tmp/nestquant/37-flash/fp8"
OUT_DEFAULT = "/tmp/nestquant/37-flash/private/dec_trace"
STOP_IDS = (154820, 154827, 154829)            # <|endoftext|>, <|user|>, <|observation|>
THINK_END = 154842                              # </think>
HOST_LIMIT_GB = 1200.0
RENAME = [(r"self_attn\.(f_a_proj|f_b_proj|dt_bias|A_log)", r"self_attn.forget_gate.\1"),
          (r"hc_attn_(fn|base|scale)", r"attn_hc.\1"), (r"hc_ffn_(fn|base|scale)", r"ffn_hc.\1")]
FP32_KEYS = ("A_log", "dt_bias", "e_score_correction_bias")
SMOKE_PROMPTS = [
    "What is 17 * 23? Answer briefly.",
    "Write a Python function that checks whether a string is a palindrome.",
    "A 62-year-old presents with sudden severe headache and neck stiffness. What is the most important first "
    "investigation, and why?",
    "Explain the difference between a fixed-rate and a variable-rate mortgage in two sentences.",
    "Under Australian contract law, what are the elements needed to form a binding contract?",
    "Prove that the square root of 2 is irrational.",
    "Summarise the causes of the 2008 financial crisis in three bullet points.",
    "Translate 'the quick brown fox jumps over the lazy dog' into French.",
]


def log(*a):
    print(time.strftime("%H:%M:%S"), "[dec37]", *a, flush=True)


# ------------------------------------------------------------------ host / device safety
def host_unreclaimable_gb():
    """anon + shmem of the box cgroup (what an OOM would have to kill; page cache is reclaimable)."""
    try:
        s = 0
        for line in open("/sys/fs/cgroup/memory.stat"):
            k, v = line.split()
            if k in ("anon", "shmem"):
                s += int(v)
        return s / 1e9
    except OSError:
        return 0.0


def rss_gb():
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1e6
    except OSError:
        pass
    return 0.0


def check_host(limit_gb, rss_limit_gb, where=""):
    u, r = host_unreclaimable_gb(), rss_gb()
    if u > limit_gb:
        raise SystemExit(f"[dec37] host unreclaimable {u:.0f} GB > {limit_gb:.0f} GB {where}: refusing to continue")
    if r > rss_limit_gb:
        raise SystemExit(f"[dec37] own RSS {r:.1f} GB > {rss_limit_gb:.0f} GB {where}: refusing to continue")
    return u, r


_TF32 = {"attn": False, "all": False}


@contextlib.contextmanager
def tf32(on):
    """matmul TF32 on/off for a region (global flag; single-threaded driver)."""
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = bool(on)
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old


# ------------------------------------------------------------------ checkpoint + dequant
E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
_LUT2 = {}


def nvfp4_unpack(packed, dtype):
    """uint8 [..., n/2] (low nibble = even element) -> e2m1 values [..., n] in dtype (exact)."""
    key = (packed.device, dtype)
    if key not in _LUT2:
        lut = torch.tensor(E2M1, dtype=torch.float32)
        b = torch.arange(256)
        _LUT2[key] = torch.stack([lut[b & 15], lut[b >> 4]], 1).to(packed.device, dtype)     # [256, 2]
    v = F.embedding(packed.reshape(-1).int(), _LUT2[key])                                   # [n/2, 2]
    return v.view(*packed.shape[:-1], packed.shape[-1] * 2)


def nvfp4_dq(packed, scale, gmul, dtype, out=None):
    """NVFP4 (16-element groups): w = e2m1 * fp8 scale * gmul.  packed [o, i/2] u8, scale [o, i/16] fp8/float,
    gmul scalar or [o] fp32 per row.  Product in fp32, one rounding to dtype."""
    o, i = packed.shape[0], packed.shape[1] * 2
    v = nvfp4_unpack(packed, torch.float32).view(o, i // 16, 16)
    g = gmul.float().view(-1, 1) if torch.is_tensor(gmul) else torch.tensor(float(gmul))
    s = scale.float() * g.to(scale.device)
    w = (v * s[..., None]).view(o, i)
    if out is None:
        return w.to(dtype)
    out.copy_(w)
    return out


class Ckpt:
    """safetensors index + streamed per-tensor reads (mmap; never more than one tensor in host buffers)."""

    def __init__(self, path):
        self.path = path
        self.idx = json.load(open(f"{path}/model.safetensors.index.json"))["weight_map"]
        self._f = {}

    def __contains__(self, k):
        return k in self.idx

    def _h(self, k):
        from safetensors import safe_open
        fn = self.idx[k]
        if fn not in self._f:
            self._f[fn] = safe_open(f"{self.path}/{fn}", "pt", device="cpu")
        return self._f[fn]

    def raw(self, k, dev):
        return self._h(k).get_tensor(k).to(dev)

    def shape(self, k):
        s = self._h(k).get_slice(k)
        return list(s.get_shape()), s.get_dtype()

    def wfmt(self, base):
        """storage format of linear `base` (key without the trailing 'weight')."""
        if base + "weight_scale_inv" in self.idx:
            return "fp8"
        if base + "weight_packed" in self.idx:
            return "nvfp4_ct"                 # compressed-tensors: weight_packed / weight_scale / weight_global_scale
        if base + "weight_scale_2" in self.idx:
            return "nvfp4_mo"                 # ModelOpt: weight (u8) / weight_scale / weight_scale_2
        return "plain"

    def nvfp4_parts(self, base, dev):
        f = self.wfmt(base)
        if f == "nvfp4_ct":
            gs = self.raw(base + "weight_global_scale", dev).float().reshape(-1)
            return self.raw(base + "weight_packed", dev), self.raw(base + "weight_scale", dev), 1.0 / gs
        return (self.raw(base + "weight", dev), self.raw(base + "weight_scale", dev),
                self.raw(base + "weight_scale_2", dev).float().reshape(-1))

    def get(self, k, dev, dtype=torch.bfloat16):
        """dequantised tensor k (fp8 128x128 blocks as capture37g.get, or NVFP4) in dtype; non-float as stored."""
        base = k[: -len("weight")] if k.endswith("weight") else None
        f = self.wfmt(base) if base is not None else "plain"
        if f == "fp8":
            w = self.raw(k, dev)
            s = self.raw(base + "weight_scale_inv", dev).float()
            s = s.repeat_interleave(128, 0).repeat_interleave(128, 1)[: w.shape[0], : w.shape[1]]
            return (w.float() * s).to(dtype)
        if f.startswith("nvfp4"):
            p, s, g = self.nvfp4_parts(base, dev)
            return nvfp4_dq(p, s, g, dtype)
        w = self.raw(k, dev)
        return w.to(dtype) if w.is_floating_point() and w.dtype != dtype else w


def dq_ref(w, s, dtype=torch.bfloat16):
    """capture37g.Experts._dq (the reference fp8 128-block dequant)."""
    return (w.float().view(s.shape[0], 128, s.shape[1], 128) * s[:, None, :, None]).view(w.shape).to(dtype)


class ExpertStore:
    """Routed experts e0..e1-1 of one MoE layer, quantised on one device (the whole layer, or one expert-parallel
    shard); dequant(b0, b1) (global expert ids) -> bf16 views [g, 2F, D], [g, D, F]."""

    def __init__(self, ck, L, cfg, dev, dtype, scratch, e0=0, e1=None):
        self.NE, self.D, self.F = cfg.n_routed_experts, cfg.hidden_size, cfg.moe_intermediate_size
        self.e0, self.e1 = e0, self.NE if e1 is None else e1
        NE, D, Fm = self.e1 - self.e0, self.D, self.F
        p = f"{PFX}layers.{L}.mlp.experts."
        self.fmt = ck.wfmt(f"{p}0.gate_proj.")
        self.dtype, self.scratch, self.dev = dtype, scratch, dev
        E = range(self.e0, self.e1)
        if self.fmt == "fp8":
            assert D % 128 == 0 and Fm % 128 == 0
            self.gu = torch.empty(NE, 2 * Fm, D, dtype=torch.float8_e4m3fn, device=dev)
            self.gus = torch.empty(NE, 2 * Fm // 128, D // 128, device=dev)
            self.dn = torch.empty(NE, D, Fm, dtype=torch.float8_e4m3fn, device=dev)
            self.dns = torch.empty(NE, D // 128, Fm // 128, device=dev)
            for j, e in enumerate(E):
                self.gu[j, :Fm] = ck.raw(f"{p}{e}.gate_proj.weight", dev)
                self.gu[j, Fm:] = ck.raw(f"{p}{e}.up_proj.weight", dev)
                self.gus[j, : Fm // 128] = ck.raw(f"{p}{e}.gate_proj.weight_scale_inv", dev)
                self.gus[j, Fm // 128:] = ck.raw(f"{p}{e}.up_proj.weight_scale_inv", dev)
                self.dn[j] = ck.raw(f"{p}{e}.down_proj.weight", dev)
                self.dns[j] = ck.raw(f"{p}{e}.down_proj.weight_scale_inv", dev)
            for t in (self.gu, self.dn):                  # e4m3 NaN codes (0x7F / 0xFF) would break the Triton dequant
                assert not ((t.view(torch.uint8) & 0x7F) == 0x7F).any(), f"L{L}: fp8 NaN codes in experts"
        elif self.fmt.startswith("nvfp4"):
            self.gu = torch.empty(NE, 2 * Fm, D // 2, dtype=torch.uint8, device=dev)
            self.gus = torch.empty(NE, 2 * Fm, D // 16, dtype=torch.float8_e4m3fn, device=dev)
            self.gug = torch.empty(NE, 2 * Fm, device=dev)
            self.dn = torch.empty(NE, D, Fm // 2, dtype=torch.uint8, device=dev)
            self.dns = torch.empty(NE, D, Fm // 16, dtype=torch.float8_e4m3fn, device=dev)
            self.dng = torch.empty(NE, D, device=dev)
            for jj, e in enumerate(E):
                for j, nm in enumerate(("gate_proj", "up_proj")):
                    pk, sc, g = ck.nvfp4_parts(f"{p}{e}.{nm}.", dev)
                    self.gu[jj, j * Fm:(j + 1) * Fm] = pk
                    self.gus[jj, j * Fm:(j + 1) * Fm] = sc.to(torch.float8_e4m3fn)
                    self.gug[jj, j * Fm:(j + 1) * Fm] = g
                pk, sc, g = ck.nvfp4_parts(f"{p}{e}.down_proj.", dev)
                self.dn[jj], self.dns[jj], self.dng[jj] = pk, sc.to(torch.float8_e4m3fn), g
        else:                                   # unquantised (tests / bf16 checkpoints)
            self.gu = torch.empty(NE, 2 * Fm, D, dtype=dtype, device=dev)
            self.dn = torch.empty(NE, D, Fm, dtype=dtype, device=dev)
            for j, e in enumerate(E):
                self.gu[j, :Fm] = ck.get(f"{p}{e}.gate_proj.weight", dev, dtype)
                self.gu[j, Fm:] = ck.get(f"{p}{e}.up_proj.weight", dev, dtype)
                self.dn[j] = ck.get(f"{p}{e}.down_proj.weight", dev, dtype)

    def nbytes(self):
        return sum(t.numel() * t.element_size() for k, t in vars(self).items()
                   if torch.is_tensor(t) and k in ("gu", "gus", "gug", "dn", "dns", "dng"))

    def dequant(self, b0, b1):
        g = b1 - b0
        e0, e1 = b0 - self.e0, b1 - self.e0
        assert 0 <= e0 and e1 <= self.e1 - self.e0, (b0, b1, self.e0, self.e1)
        if self.fmt == "plain":
            return self.gu[e0:e1], self.dn[e0:e1]
        ogu, odn = self.scratch.get(g, self.D, self.F, self.dtype)
        if self.fmt == "fp8":
            if _DQ["triton"] and self.dtype == torch.bfloat16:
                fp8_dq_triton(self.gu[e0:e1], self.gus[e0:e1], ogu)
                fp8_dq_triton(self.dn[e0:e1], self.dns[e0:e1], odn)
            else:
                ogu.copy_(self.gu[e0:e1])                                         # fp8 -> bf16 exact
                ogu.view(g, self.gus.shape[1], 128, self.gus.shape[2], 128).mul_(self.gus[e0:e1, :, None, :, None])
                odn.copy_(self.dn[e0:e1])
                odn.view(g, self.dns.shape[1], 128, self.dns.shape[2], 128).mul_(self.dns[e0:e1, :, None, :, None])
        else:
            for j, e in enumerate(range(e0, e1)):
                nvfp4_dq(self.gu[e], self.gus[e], self.gug[e], self.dtype, out=ogu[j])
                nvfp4_dq(self.dn[e], self.dns[e], self.dng[e], self.dtype, out=odn[j])
        return ogu, odn

    def ref(self, e):
        """reference dequant of (global) expert e (fp8: capture37g _dq) -> (gu, dn)."""
        e = e - self.e0
        if self.fmt == "fp8":
            return dq_ref(self.gu[e], self.gus[e], self.dtype), dq_ref(self.dn[e], self.dns[e], self.dtype)
        if self.fmt == "plain":
            return self.gu[e], self.dn[e]
        return (nvfp4_dq(self.gu[e], self.gus[e], self.gug[e], self.dtype),
                nvfp4_dq(self.dn[e], self.dns[e], self.dng[e], self.dtype))


# ------------------------------------------------------------------ fused fp8 block dequant (Triton, sm_80-safe)
_DQ = {"triton": False}
_TRI = {}


def _triton_kernel():
    """one-pass fp8(e4m3, read as uint8) x block-128 fp32 scale -> bf16. sm_80 Triton cannot touch fp8 types, so the
    e4m3 bits are decoded with integer ops (normals: re-biased exponent; subnormals: m * 2^-9). Result is bitwise
    = copy_(fp8 -> bf16) + mul_(fp32 scale) (both are RNE(fp32(w) * s)); verified at start by Engine.selfcheck."""
    if "k" in _TRI:
        return _TRI["k"]
    import triton
    import triton.language as tl

    @triton.jit
    def _k(W, S, O, R, C, RB, CB, BR: tl.constexpr, BC: tl.constexpr):
        e = tl.program_id(0).to(tl.int64)
        rt = tl.program_id(1)
        ct = tl.program_id(2)
        r = rt * BR + tl.arange(0, BR)[:, None]
        c = ct * BC + tl.arange(0, BC)[None, :]
        off = e * R * C + r.to(tl.int64) * C + c
        b = tl.load(W + off).to(tl.uint32)
        sg = (b >> 7) & 1
        ex = (b >> 3) & 15
        m = b & 7
        v = ((sg << 31) | ((ex + 120) << 23) | (m << 20)).to(tl.float32, bitcast=True)
        sub = ((m.to(tl.float32) * 0.001953125).to(tl.uint32, bitcast=True) | (sg << 31)).to(tl.float32, bitcast=True)
        v = tl.where(ex == 0, sub, v)                               # subnormal m * 2^-9, sign incl. -0.0
        sc = tl.load(S + e * RB * CB + ((rt * BR) // 128) * CB + ct)
        x = (v * sc).to(tl.uint32, bitcast=True)                  # explicit RNE fp32 -> bf16 (finite inputs):
        x = (x + 0x7FFF + ((x >> 16) & 1)) >> 16                    # identical compiled / interpreted (the
        tl.store(O + off, x.to(tl.uint16).to(tl.int16, bitcast=True))  # interpreter's .to(bf16) truncates)

    _TRI["k"] = _k
    return _k


def fp8_dq_triton(w, s, out):
    """w [g, R, C] float8_e4m3fn, s [g, R/128, C/128] fp32, out [g, R, C] bf16 (all contiguous)."""
    g, R, C = w.shape
    assert w.is_contiguous() and s.is_contiguous() and out.is_contiguous() and R % 128 == 0 and C % 128 == 0
    BR = 32
    _triton_kernel()[(g, R // BR, C // 128)](w.view(torch.uint8), s, out.view(torch.int16), R, C, R // 128, C // 128, BR=BR, BC=128,
                                              num_warps=4)
    return out


class Scratch:
    """per-device bf16 dequant scratch for G experts."""

    def __init__(self, G, D, Fm, dtype, dev):
        self.G = G
        self.gu = torch.empty(G, 2 * Fm, D, dtype=dtype, device=dev)
        self.dn = torch.empty(G, D, Fm, dtype=dtype, device=dev)

    def get(self, g, D, Fm, dtype):
        assert g <= self.G and self.gu.dtype == dtype
        return self.gu[:g], self.dn[:g]


# ------------------------------------------------------------------ one decoder layer: HF modules + our caches
def build_hf_layer(M, ck, cfg, L, dev, dtype):
    """= capture37g.build_layer with dtype for the bf16 tensors and NVFP4-aware dequant; experts -> Identity."""
    with torch.device("meta"):
        lay = M.Glm5NextTextDecoderLayer(cfg, L)
    p = f"{PFX}layers.{L}."
    sd, conv = {}, {}
    skip = ("weight_scale_inv", "weight_scale", "weight_scale_2", "weight_global_scale", "input_scale",
            "input_global_scale")
    for k in ck.idx:
        if not k.startswith(p) or ".mlp.experts." in k or k.endswith(skip):
            continue
        r = k[len(p):]
        if r.endswith("weight_packed"):
            r, k = r[: -len("_packed")], k[: -len("_packed")]
        m = re.match(r"self_attn\.([qkv])_conv1d\.weight", r)
        if m:
            conv[m.group(1)] = ck.get(k, dev, dtype)
            continue
        for a, b in RENAME:
            r = re.sub(a, b, r)
        fp32 = r.endswith(FP32_KEYS) or "_hc." in r
        sd[r] = ck.get(k, dev, torch.float32 if fp32 else dtype)
    if conv:
        sd["self_attn.conv1d.weight"] = torch.cat([conv["q"], conv["k"], conv["v"]], 0)
    if cfg.mlp_layer_types[L] == "sparse":
        lay.mlp.experts = torch.nn.Identity()
    missing, unexpected = lay.load_state_dict(sd, strict=False, assign=True)
    assert not unexpected and not missing, (L, missing, unexpected)
    for n, b in lay.named_buffers():
        assert b.device.type != "meta", (L, n)
    return lay.eval()


class Shard:
    """the routed experts e0..e1-1 of one layer on one device, as contiguous expert blocks [(b0, b1)]."""

    def __init__(self, dev, store, blocks, e0, e1, remote):
        self.dev, self.store, self.blocks, self.e0, self.e1, self.remote = dev, store, blocks, e0, e1, remote


class ExpertSet:
    """per-layer view over the shards: fmt / ref(e) / dequant of a block (self-check, tests)."""

    def __init__(self, shards):
        self.shards = shards
        self.fmt = shards[0].store.fmt

    def shard_of(self, e):
        return next(sh for sh in self.shards if sh.e0 <= e < sh.e1)

    def ref(self, e):
        return self.shard_of(e).store.ref(e)

    def dequant(self, b0, b1):
        return self.shard_of(b0).store.dequant(b0, b1)


class Layer:
    def __init__(self, eng, L, dev):
        self.eng, self.L, self.dev = eng, L, dev
        cfg, dt = eng.cfg, eng.dtype
        self.lay = build_hf_layer(eng.M, eng.ck, cfg, L, dev, dt)
        self.kda = cfg.layer_types[L] == "linear_attention"
        self.sparse = cfg.mlp_layer_types[L] == "sparse"
        a = self.lay.self_attn
        if self.kda:
            self.nh, self.hd = a.num_heads, a.head_dim
            self.C = a.conv_dim
            self.convw = a.conv1d.weight[:, 0, :].t().float().contiguous()             # [K, C] fp32
            self.ck = a.conv_kernel_size
        else:
            assert cfg.qk_rope_head_dim == 0 and a.indexer is not None, "dec37 assumes NoPE MLA + full indexers"
            self.nh = a.num_heads
            self.dn, self.dv, self.r = a.qk_nope_head_dim, a.v_head_dim, a.kv_lora_rank
            kvb = a.kv_b_proj.weight.view(self.nh, self.dn + self.dv, self.r)
            self.W_uk = kvb[:, : self.dn].contiguous()                                # [H, dn, r]
            self.W_uvT = kvb[:, self.dn:].transpose(1, 2).contiguous()                  # [H, r, dv]
            ind = a.indexer
            self.kp, self.nsel = ind.index_kpool, ind.index_topk // ind.index_kpool
            self.ape = ind.index_kpool_compress_ape.float()
            self.ih, self.ihd = ind.n_heads, ind.head_dim
            assert ind.index_kpool_always_select_tail
        self.shards, self.ex = [], None
        if self.sparse:
            self.NE = cfg.n_routed_experts
            plan = eng.expert_plan(L, dev) if hasattr(eng, "expert_plan") else \
                [(dev, [(b0, min(self.NE, b0 + eng.G)) for b0 in range(0, self.NE, eng.G)], False)]
            for sdev, blocks, remote in plan:
                e0, e1 = blocks[0][0], blocks[-1][1]
                st = ExpertStore(eng.ck, L, cfg, sdev, dt, eng.scratch[sdev], e0, e1)
                self.shards.append(Shard(sdev, st, blocks, e0, e1, remote))
            self.ex = ExpertSet(self.shards)

    # ---------------------------------------------------------------- caches
    def state_bytes_per_slot(self, Smax):
        return Layer.slot_bytes(self.eng.cfg, self.L, Smax)

    @staticmethod
    def slot_bytes(cfg, L, Smax):
        b = 0
        if cfg.layer_types[L] == "linear_attention":
            H, d = cfg.linear_num_heads, cfg.linear_head_dim
            b += H * d * d * 4 + (cfg.linear_conv_kernel_dim - 1) * 3 * H * d * 2
        else:
            b += Smax * cfg.kv_lora_rank * 2 + (Smax // cfg.index_kpool + 2) * cfg.index_head_dim * 2 \
                + 2 * cfg.index_kpool * cfg.index_head_dim * 2
        if cfg.mlp_layer_types[L] == "sparse":
            b += Smax * cfg.num_experts_per_tok * 4 + Smax * 4
        return b

    def alloc(self, NS, Smax):
        dev, dt, cfg = self.dev, self.eng.dtype, self.eng.cfg
        if self.kda:
            self.S = torch.zeros(NS, self.nh, self.hd, self.hd, dtype=torch.float32, device=dev)
            self.conv = torch.zeros(NS, self.ck - 1, self.C, dtype=dt, device=dev)
        else:
            self.Cl = torch.zeros(NS, Smax, self.r, dtype=dt, device=dev)
            self.npool = Smax // self.kp + 2
            self.PK = torch.zeros(NS, self.npool, self.ihd, dtype=dt, device=dev)
            self.KR = torch.zeros(NS, self.kp, self.ihd, dtype=dt, device=dev)
            self.GR = torch.zeros(NS, self.kp, self.ihd, dtype=dt, device=dev)
        if self.sparse:
            K = cfg.num_experts_per_tok
            self.RI = torch.zeros(NS, Smax, K, dtype=torch.int16, device=dev)
            self.RW = torch.zeros(NS, Smax, K, dtype=torch.float16, device=dev)
            self.RX = torch.zeros(NS, Smax, dtype=torch.float32, device=dev)

    def reset_slot(self, s):
        if self.kda:
            self.S[s].zero_()
            self.conv[s].zero_()

    def records(self, s, n):
        return self.RI[s, :n], self.RW[s, :n], self.RX[s, :n]

    # ---------------------------------------------------------------- MoE / MLP
    def mlp(self, x, rec):
        lay = self.lay
        if not self.sparse:
            return lay.mlp(x)
        with tf32(_TF32["all"]):
            _, tw, ti = lay.mlp.gate(x)                                            # HF router (fp32)
        if rec is not None:
            sl, ps = rec
            self.RI[sl, ps] = ti.to(torch.int16)
            self.RW[sl, ps] = tw.to(torch.float16)
            self.RX[sl, ps] = x.float().square().sum(1)
        y = self.experts(x, ti, tw)
        return (y + lay.mlp.shared_experts(x).float()).to(x.dtype)

    def experts(self, x, ti, tw):
        """routed experts. Each (token, k) assignment gets its own fp32 output row (no index_add collisions), the
        K rows of a token are summed in router order at the end: deterministic, and bitwise independent of how
        the expert blocks are spread over devices (expert-parallel shards) for a fixed --expert-block."""
        NE, K = self.NE, ti.shape[1]
        n, D = x.shape
        lim = self.eng.cfg.swiglu_limit
        flat = ti.reshape(-1)
        order = torch.argsort(flat, stable=True)
        cnt = torch.bincount(flat, minlength=NE)
        cnt_h = cnt.tolist()                                                       # the one host sync per layer
        starts_h = np.r_[0, np.cumsum(cnt_h)]
        pw = tw.reshape(-1)[order].float()
        starts_d = torch.cumsum(cnt, 0) - cnt
        Ys = torch.empty(n * K, D, dtype=torch.float32, device=x.device)          # per assignment, expert-sorted

        def swiglu(gu):
            g, u = gu.chunk(2, -1)
            return F.silu(g.clamp(max=lim)) * u.clamp(-lim, lim)

        for sh in self.shards:
            lo, hi = int(starts_h[sh.e0]), int(starts_h[sh.e1])
            if hi == lo:
                continue
            dv = sh.dev
            if sh.remote:                                    # async peer copies (stream-ordered by PyTorch)
                xd = x.to(dv, non_blocking=True, copy=True)
                od = order[lo:hi].to(dv, non_blocking=True, copy=True)
                pwd = pw[lo:hi].to(dv, non_blocking=True, copy=True)
                Yd = torch.empty(hi - lo, D, dtype=torch.float32, device=dv)
            else:
                xd, od, pwd, Yd = x, order[lo:hi], pw[lo:hi], Ys[lo:hi]
            rows = od // K
            fl = flat.to(dv, non_blocking=True, copy=True) if sh.remote else flat
            sd = starts_d.to(dv, non_blocking=True, copy=True) if sh.remote else starts_d
            esort = fl[od]
            for b0, b1 in sh.blocks:
                s0, s1 = int(starts_h[b0]), int(starts_h[b1])
                if s1 == s0:
                    continue
                gu, dn = sh.store.dequant(b0, b1)
                g = b1 - b0
                maxc = max(cnt_h[b0:b1])
                a0, a1 = s0 - lo, s1 - lo
                r = rows[a0:a1]
                if maxc * g <= 2 * (s1 - s0) + 32 * g:                            # padded bmm (decode)
                    e = esort[a0:a1]
                    dst = (e - b0) * maxc + (torch.arange(s0, s1, device=dv) - sd[e])
                    Xp = xd.new_zeros(g * maxc, D)
                    Xp[dst] = xd[r]
                    h = swiglu(torch.bmm(Xp.view(g, maxc, D), gu.transpose(1, 2)))
                    y = torch.bmm(h, dn.transpose(1, 2)).view(g * maxc, D)[dst]
                else:                                                              # per expert (prefill)
                    ys = []
                    for j in range(g):
                        c0, c1 = int(starts_h[b0 + j]) - lo, int(starts_h[b0 + j + 1]) - lo
                        if c1 > c0:
                            ys.append(F.linear(swiglu(F.linear(xd[rows[c0:c1]], gu[j])), dn[j]))
                    y = torch.cat(ys)
                Yd[a0:a1] = y.float() * pwd[a0:a1, None]
            if sh.remote:
                Ys[lo:hi].copy_(Yd, non_blocking=True)
        Y = torch.empty_like(Ys)
        Y[order] = Ys                                                              # unique targets: deterministic
        return Y.view(n, K, D).sum(1)

    # ---------------------------------------------------------------- attention: decode (all NS slots, 1 token)
    def attn_decode(self, hs, st):
        return self.kda_decode(hs, st) if self.kda else self.dsa_decode(hs, st)

    def kda_decode(self, hs, st):
        a = self.lay.self_attn
        n, H, d = hs.shape[0], self.nh, self.hd
        mixed = torch.cat([a.q_proj(hs), a.k_proj(hs), a.v_proj(hs)], -1)              # [n, C]
        win = torch.cat([self.conv, mixed[:, None]], 1)                                # [n, K, C]
        self.conv.copy_(win[:, 1:])
        co = F.silu((win.float() * self.convw[None]).sum(1).to(mixed.dtype))
        q, k, v = co.view(n, 3, H, d).unbind(1)
        g = a.forget_gate(hs[None])[0]                                                 # [n, H, d] fp32
        beta = torch.sigmoid(a.b_proj(hs))                                             # [n, H]
        with tf32(False):
            q = self.eng.M.l2norm(q.float(), dim=-1, eps=1e-6) * (d ** -0.5)
            k = self.eng.M.l2norm(k.float(), dim=-1, eps=1e-6)
            S = self.S
            S.mul_(g.exp()[..., None])
            kv = torch.matmul(k[:, :, None, :], S)[:, :, 0]                            # [n, H, d]
            delta = (v.float() - kv) * beta.float()[..., None]
            S.view(n * H, d, d).baddbmm_(k.reshape(n * H, d, 1), delta.reshape(n * H, 1, d))
            o = torch.matmul(q[:, :, None, :], S)[:, :, 0]
        gate = a.g_b_proj(a.g_a_proj(hs)).view(n, H, d)
        o = a.o_norm(o.to(hs.dtype), gate).reshape(n, -1)
        return a.o_proj(o)

    def pool(self, k4, g4):
        """kpool-4 pooled indexer key, HF get_pooled_states op order: softmax over tokens of gate+ape, weighted sum."""
        p = (g4.float() + self.ape).softmax(dim=-2).to(k4.dtype)
        return (p * k4).sum(dim=-2)

    def index_scores(self, qi, wts, pk):
        """qi [r, ih, ihd], wts [r, ih] fp32, pk [r, P, ihd] or [P, ihd] -> [r, P] fp32 (HF indexer scoring)."""
        ind = self.lay.self_attn.indexer
        pkT = pk.float().transpose(-1, -2)
        with tf32(_TF32["attn"]):
            s = F.relu(torch.matmul(qi.float(), pkT) * ind.softmax_scale)
            return torch.matmul(wts[:, None, :], s)[:, 0]

    def dsa_q(self, x):
        a = self.lay.self_attn
        ind = a.indexer
        qr = a.q_a_layernorm(a.q_a_proj(x))
        q = a.q_b_proj(qr).view(x.shape[0], self.nh, self.dn)
        qlat = torch.bmm(q.transpose(0, 1), self.W_uk).transpose(0, 1)                # [n, H, r]
        cl = a.kv_a_layernorm(a.kv_a_proj_with_mqa(x))
        k = ind.k_norm(ind.wk(x))
        gs = F.linear(x, ind.index_kpool_compress_gate)
        qi = ind.wq_b(qr).view(x.shape[0], self.ih, self.ihd)
        wts = ind.weights_proj(x).float() * (self.ih ** -0.5)
        return qlat, cl, k, gs, qi, wts

    def attend(self, qlat, Ck, mask):
        """qlat [r, H, R], Ck [r or 1, T, R] latent keys = values, mask [r, T] bool -> o_proj input [r, H*dv]."""
        a = self.lay.self_attn
        with tf32(_TF32["attn"]):
            s = torch.matmul(qlat, Ck.transpose(-1, -2)) * a.scaling                   # [r, H, T]
            s = s.masked_fill(~mask[:, None, :], float("-inf"))
            p = torch.softmax(s, dim=-1, dtype=torch.float32).to(qlat.dtype)
            ol = torch.matmul(p, Ck)                                                   # [r, H, R]
            o = torch.bmm(ol.transpose(0, 1), self.W_uvT).transpose(0, 1)              # [r, H, dv]
        return o.reshape(o.shape[0], -1)

    def sparse_mask(self, qi, wts, pk, pos, T):
        """token mask [r, T] for rows whose visible pools exceed nsel: top-nsel pools (of the visible) + tail."""
        nvis = (pos + 1) // self.kp
        Pm = pk.shape[-2]
        sc = self.index_scores(qi, wts, pk)
        ar = torch.arange(Pm, device=pos.device)
        vis = ar[None] < nvis[:, None]
        sc = sc.masked_fill(~vis, torch.finfo(sc.dtype).min)                         # = HF fill (tie parity)
        top = sc.topk(min(self.nsel, Pm), dim=-1).indices
        sel = torch.zeros_like(vis).scatter_(1, top, True) & vis
        tok = sel.repeat_interleave(self.kp, 1)
        if tok.shape[1] < T:
            tok = F.pad(tok, (0, T - tok.shape[1]))
        tok = tok[:, :T]
        at = torch.arange(T, device=pos.device)[None]
        tail = (at >= self.kp * nvis[:, None]) & (at <= pos[:, None])
        return tok | tail

    def dsa_decode(self, hs, st):
        a = self.lay.self_attn
        n = hs.shape[0]
        qlat, cl, k, gs, qi, wts = self.dsa_q(hs)
        pos = st.pos[self.dev]
        ar = torch.arange(n, device=hs.device)
        self.Cl[ar, pos] = cl
        j = pos % self.kp
        self.KR[ar, j] = k
        self.GR[ar, j] = gs
        pk_new = self.pool(self.KR, self.GR)
        pidx = torch.where(j == self.kp - 1, pos // self.kp, torch.full_like(pos, self.npool - 1))
        self.PK[ar, pidx] = pk_new
        out = torch.empty(n, self.nh * self.dv, dtype=hs.dtype, device=hs.device)
        lens_h = st.pos_h + 1
        nvis_h = lens_h // self.kp
        H, R = self.nh, self.r
        budget = self.eng.attn_budget
        c0 = 0
        while c0 < n:
            Tm0 = int(lens_h[c0])
            rmax = max(1, budget // (H * max(Tm0, 1) * 12))
            c1 = min(n, c0 + rmax)
            Tm = int(lens_h[c0:c1].max())
            while c1 - c0 > 1 and (c1 - c0) * H * Tm * 12 > budget:
                c1 = c0 + max(1, (c1 - c0) // 2)
                Tm = int(lens_h[c0:c1].max())
            p = pos[c0:c1]
            mask = torch.arange(Tm, device=hs.device)[None] <= p[:, None]
            if (nvis_h[c0:c1] > self.nsel).any():
                Pm = int(nvis_h[c0:c1].max())
                sm = self.sparse_mask(qi[c0:c1], wts[c0:c1], self.PK[c0:c1, :Pm], p, Tm)
                spr = (p + 1) // self.kp > self.nsel
                mask = torch.where(spr[:, None], sm & mask, mask)
            out[c0:c1] = self.attend(qlat[c0:c1], self.Cl[c0:c1, :Tm], mask)
            c0 = c1
        return a.o_proj(out)

    # ---------------------------------------------------------------- attention: prefill (a wave of sequences)
    def attn_prefill(self, hs, wv):
        return self.kda_prefill(hs, wv) if self.kda else self.dsa_prefill(hs, wv)

    def kda_prefill(self, hs, wv):
        a, M = self.lay.self_attn, self.eng.M
        H, d = self.nh, self.hd
        out = torch.empty_like(hs)
        piece = self.eng.kda_piece
        bmax = max(1, self.eng.kda_tokens // piece)
        order = sorted(range(len(wv.T)), key=lambda i: -wv.T[i])
        for gi in range(0, len(order), bmax):
            grp = order[gi:gi + bmax]
            Ts = [wv.T[i] for i in grp]
            Lp = -(-max(Ts) // piece) * piece
            b = len(grp)
            xs = hs.new_zeros(b, Lp, hs.shape[1])
            for j, i in enumerate(grp):
                xs[j, : Ts[j]] = hs[wv.off[i]: wv.off[i] + Ts[j]]
            mixed = torch.cat([a.q_proj(xs), a.k_proj(xs), a.v_proj(xs)], -1)              # [b, Lp, C]
            for j, i in enumerate(grp):
                s, T = wv.slot[i], Ts[j]
                self.conv[s].zero_()
                w = min(T, self.ck - 1)
                self.conv[s, self.ck - 1 - w:] = mixed[j, T - w:T]
            co = M.causal_conv1d_fn(mixed.transpose(1, 2), weight=a.conv1d.weight.squeeze(1), bias=None,
                                    activation=a.activation).transpose(1, 2)
            del mixed
            valid = torch.arange(Lp, device=hs.device)[None] < torch.tensor(Ts, device=hs.device)[:, None]
            co = co * valid[..., None].to(co.dtype)
            q, k, v = [t.reshape(b, Lp, H, d) for t in co.split(self.C // 3, -1)]
            g = a.forget_gate(xs) * valid[..., None, None]
            beta = torch.sigmoid(a.b_proj(xs)) * valid[..., None].to(xs.dtype)
            state = torch.zeros(b, H, d, d, dtype=torch.float32, device=hs.device)
            core = torch.empty(b, Lp, H, d, dtype=hs.dtype, device=hs.device)
            with tf32(_TF32["all"]):
                for p0 in range(0, Lp, piece):
                    p1 = p0 + piece
                    if p0 >= max(Ts):
                        break
                    o, state = M.chunk_kimi_delta_attention(q[:, p0:p1], k[:, p0:p1], v[:, p0:p1], g=g[:, p0:p1],
                                                            beta=beta[:, p0:p1], chunk_size=64, initial_state=state,
                                                            output_final_state=True, use_qk_l2norm_in_kernel=True)
                    core[:, p0:p1] = o
            for j, i in enumerate(grp):
                self.S[wv.slot[i]] = state[j].float()
            gate = a.g_b_proj(a.g_a_proj(xs)).view(b, Lp, H, d)
            o = a.o_norm(core, gate).reshape(b, Lp, -1)
            o = a.o_proj(o)
            for j, i in enumerate(grp):
                out[wv.off[i]: wv.off[i] + Ts[j]] = o[j, : Ts[j]]
        return out

    def dsa_prefill(self, hs, wv):
        a = self.lay.self_attn
        out = torch.empty(hs.shape[0], self.nh * self.dv, dtype=hs.dtype, device=hs.device)
        H = self.nh
        for i in range(len(wv.T)):
            s, T, o0 = wv.slot[i], wv.T[i], wv.off[i]
            x = hs[o0:o0 + T]
            qlat, cl, k, gs, qi, wts = self.dsa_q(x)
            self.Cl[s, :T] = cl
            P = T // self.kp
            if P:
                self.PK[s, :P] = self.pool(k[: P * self.kp].view(P, self.kp, -1), gs[: P * self.kp].view(P, self.kp, -1))
            rem = T - P * self.kp
            if rem:
                self.KR[s, :rem] = k[P * self.kp:]
                self.GR[s, :rem] = gs[P * self.kp:]
            Ck = self.Cl[s, :T]
            t0 = 0
            while t0 < T:
                c = max(1, self.eng.attn_budget // (H * T * 12))
                t1 = min(T, t0 + c)
                pos = torch.arange(t0, t1, device=hs.device)
                Tk = t1
                mask = torch.arange(Tk, device=hs.device)[None] <= pos[:, None]
                if t1 // self.kp > self.nsel:                       # some rows see > nsel complete pools
                    Pm = T // self.kp                               # HF no-cache width (all complete pools): tie parity
                    sm = self.sparse_mask(qi[t0:t1], wts[t0:t1], self.PK[s, :Pm], pos, Tk)
                    spr = (pos + 1) // self.kp > self.nsel
                    mask = torch.where(spr[:, None], sm & mask, mask)
                out[o0 + t0:o0 + t1] = self.attend(qlat[t0:t1], Ck[None, :Tk], mask)
                t0 = t1
        return a.o_proj(out)

    # ---------------------------------------------------------------- full layer
    def hc_in(self, hc, norm, H):
        post, comb, hs = hc(H[None])
        return post[0], comb[0], norm(hs)[0]

    @staticmethod
    def hc_out(post, comb, o, H):
        dt = H.dtype
        return post.to(dt).unsqueeze(-1) * o.unsqueeze(-2) + torch.matmul(comb.to(dt).transpose(-1, -2), H)

    def forward_decode(self, H, st):
        lay = self.lay
        post, comb, hs = self.hc_in(lay.attn_hc, lay.input_layernorm, H)
        H = self.hc_out(post, comb, self.attn_decode(hs, st), H)
        post, comb, x = self.hc_in(lay.ffn_hc, lay.post_attention_layernorm, H)
        rec = (st.ar[self.dev], st.pos[self.dev]) if self.sparse else None
        return self.hc_out(post, comb, self.mlp(x, rec), H)

    def forward_prefill(self, H, wv):
        lay = self.lay
        n, ch = H.shape[0], self.eng.tok_chunk
        hs = torch.empty(n, H.shape[-1], dtype=H.dtype, device=H.device)
        for c0 in range(0, n, ch):
            _, _, hs[c0:c0 + ch] = self.hc_in(lay.attn_hc, lay.input_layernorm, H[c0:c0 + ch])
        o = self.attn_prefill(hs, wv)
        del hs
        Hn = torch.empty_like(H)
        for c0 in range(0, n, ch):
            post, comb, _ = lay.attn_hc(H[None, c0:c0 + ch])
            Hn[c0:c0 + ch] = self.hc_out(post[0], comb[0], o[c0:c0 + ch], H[c0:c0 + ch])
        del o
        sl, ps = wv.rec(self.dev)
        for c0 in range(0, n, ch):
            h = Hn[c0:c0 + ch]
            post, comb, x = self.hc_in(lay.ffn_hc, lay.post_attention_layernorm, h)
            rec = (sl[c0:c0 + ch], ps[c0:c0 + ch]) if self.sparse else None
            Hn[c0:c0 + ch] = self.hc_out(post, comb, self.mlp(x, rec), h)
        return Hn


# ------------------------------------------------------------------ model / pipeline
class Wave:
    """a prefill wave: sequences (slot, tokens) concatenated."""

    def __init__(self, slots, toks):
        self.slot, self.T = list(slots), [len(t) for t in toks]
        self.off = list(np.r_[0, np.cumsum(self.T)[:-1]].astype(int)) if toks else []
        self.N = int(sum(self.T))
        self.tok = torch.from_numpy(np.concatenate(toks).astype(np.int64))
        self._rec = {}

    def rec(self, dev):
        if dev not in self._rec:
            sl = torch.cat([torch.full((t,), s, dtype=torch.long) for s, t in zip(self.slot, self.T)])
            ps = torch.cat([torch.arange(t) for t in self.T])
            self._rec[dev] = (sl.to(dev), ps.to(dev))
        return self._rec[dev]

    def last_rows(self):
        return [o + t - 1 for o, t in zip(self.off, self.T)]


class StepState:
    def __init__(self, NS, devs):
        self.NS = NS
        self.pos_h = np.zeros(NS, np.int64)
        self.pos = {}
        self.ar = {d: torch.arange(NS, device=d) for d in devs}
        self.devs = devs

    def push(self):
        t = torch.from_numpy(self.pos_h)
        for d in self.devs:
            self.pos[d] = t.to(d)


def plan_partition(costs, ndev):
    """contiguous split of per-layer costs over ndev devices minimising the max device load."""
    lo, hi = max(costs), sum(costs)

    def fits(cap):
        parts, cur, n = [], 0, 1
        for c in costs:
            if cur + c > cap:
                n += 1
                cur = 0
            cur += c
        return n <= ndev

    while hi - lo > 1e6:
        mid = (lo + hi) / 2
        if fits(mid):
            hi = mid
        else:
            lo = mid
    assign, cur, d = [], 0, 0
    for i, c in enumerate(costs):
        left = len(costs) - i
        if (cur + c > hi and cur > 0) or (ndev - d - 1 >= left and cur > 0 and d < ndev - 1):
            d += 1
            cur = 0
        assign.append(d)
        cur += c
    return assign


class Engine:
    def __init__(self, a, NS, Smax):
        from transformers import AutoConfig
        from transformers.models.glm5_next import modeling_glm5_next as M
        self.M, self.a = M, a
        self.ck = Ckpt(a.ckpt)
        cfg_full = AutoConfig.from_pretrained(a.ckpt)
        cfg = getattr(cfg_full, "text_config", cfg_full)
        cfg.num_local_experts = cfg.n_routed_experts
        self.cfg = cfg
        assert all(t == "full" for t in cfg.indexer_types), "shared indexers not supported"
        assert cfg.n_group == 1, cfg.n_group
        self.dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[a.dtype]
        self.devs = [torch.device(d) for d in a.devices.split(",")]
        self.NS, self.Smax = NS, Smax
        NE = cfg.n_routed_experts
        ndev = len(self.devs)
        self.ep = bool(a.ep) and ndev > 1
        self.G = a.expert_block or (-(-NE // ndev) if self.ep else 16)
        self.tok_chunk = a.tok_chunk
        self.kda_piece, self.kda_tokens = a.kda_piece, a.kda_tokens
        self.attn_budget = int(a.attn_budget_gb * 2**30)
        nL = cfg.num_hidden_layers
        self.nL = nL
        # ---- expert blocks (fixed [k*G, (k+1)*G) partition in every mode) and their device shards for --ep
        blocks = [(b0, min(NE, b0 + self.G)) for b0 in range(0, NE, self.G)]
        self.ep_blocks = [blocks[round(j * len(blocks) / ndev):round((j + 1) * len(blocks) / ndev)]
                          for j in range(ndev)] if self.ep else None
        # ---- per-layer bytes (weights from the safetensors headers + slot state)
        wb, xb = self.layer_weight_bytes()
        sb = [Layer.slot_bytes(cfg, L, Smax) * NS for L in range(nL)]
        costs = [wb[L] + sb[L] + (0 if self.ep else xb[L]) for L in range(nL)]
        V, D = cfg.vocab_size, cfg.hidden_size
        eb = 2 * V * D * self.dtype.itemsize // 2 * 1                     # embed (first) / head (last) + logits ws
        logits_ws = NS * V * 4 * 6
        costs_adj = list(costs)
        costs_adj[0] += eb
        costs_adj[-1] += eb + logits_ws
        self.place = plan_partition(costs_adj, ndev) if a.place == "auto" else [int(x) for x in a.place.split(",")]
        assert len(self.place) == nL
        per = [0] * ndev
        for L in range(nL):
            per[self.place[L]] += costs_adj[L]
        if self.ep:
            for j in range(ndev):
                ne_j = sum(b1 - b0 for b0, b1 in self.ep_blocks[j])
                per[j] += sum(xb) * ne_j // NE
        sc_b = self.G * 3 * cfg.moe_intermediate_size * D * self.dtype.itemsize
        self.dev_of = [self.devs[self.place[L]] for L in range(nL)]
        log(f"plan NS {NS} Smax {Smax}: per-layer weights {(sum(wb) + sum(xb)) / 1e9:.1f} GB (experts {sum(xb) / 1e9:.1f}), "
            f"state {sum(sb) / 1e9:.1f} GB; expert block {self.G}, "
            + (f"expert-parallel over {ndev} devices ({[len(b) for b in self.ep_blocks]} blocks); " if self.ep else "")
            + "devices " + " ".join(f"{self.devs[d]}:{per[d] / 1e9:.1f}GB(L{self.place.index(d)}-"
                                    f"L{nL - 1 - self.place[::-1].index(d)})" for d in range(ndev) if d in self.place))
        reserve = a.reserve_gb * 2**30
        for d in range(ndev):
            dev = self.devs[d]
            if dev.type != "cuda":
                continue
            free, total = torch.cuda.mem_get_info(dev)
            need = per[d] + sc_b + reserve
            if need > free * 0.98:
                raise SystemExit(f"[dec37] {dev}: planned {per[d] / 1e9:.1f} GB + scratch {sc_b / 1e9:.1f} + reserve "
                                 f"{reserve / 1e9:.0f} GB > free {free / 1e9:.1f} GB: lower --slots / --smax")
            torch.cuda.set_per_process_memory_fraction(min(0.99, (need + 4 * 2**30) / total), dev)
        _DQ["triton"] = False
        if a.dq in ("auto", "triton") and self.dtype == torch.bfloat16 and all(d.type == "cuda" for d in self.devs):
            try:
                _triton_kernel()
                _DQ["triton"] = True
            except Exception as ex:                                            # noqa: BLE001
                if a.dq == "triton":
                    raise
                log(f"triton dequant unavailable ({ex!r}); torch dequant")
        elif a.dq == "triton" and os.environ.get("TRITON_INTERPRET") == "1" and self.dtype == torch.bfloat16:
            _DQ["triton"] = True                                               # CPU tests (interpreter)
        sdevs = set(self.devs) if self.ep else set(self.dev_of)
        self.scratch = {dev: Scratch(self.G, D, cfg.moe_intermediate_size, self.dtype, dev) for dev in sdevs}
        # ---- load
        t0 = time.time()
        d0, dl = self.dev_of[0], self.dev_of[-1]
        self.embed = self.ck.get(PFX + "embed_tokens.weight", d0, self.dtype)
        self.layers = []
        for L in range(nL):
            check_host(a.host_limit_gb, a.rss_limit_gb, f"loading L{L}")
            self.layers.append(Layer(self, L, self.dev_of[L]))
            self.layers[-1].alloc(NS, Smax)
            if L % 5 == 0 or L == nL - 1:
                log(f"loaded L{L} on {self.dev_of[L]} ({time.time() - t0:.0f}s, host RSS {rss_gb():.1f} GB)")
        self.norm = M.Glm5NextTextRMSNorm(D, cfg.rms_norm_eps).to(dl)
        self.norm.weight.data = self.ck.get(PFX + "norm.weight", dl, torch.float32).to(self.dtype)
        self.head = self.ck.get("lm_head.weight", dl, self.dtype)
        self.hh = M.Glm5NextTextHyperHead()
        self.selfcheck()
        for dev in set(self.dev_of):
            if dev.type == "cuda":
                log(f"{dev}: allocated {torch.cuda.memory_allocated(dev) / 1e9:.1f} GB")

    def layer_weight_bytes(self):
        """-> (non-expert bytes per layer as loaded, routed-expert bytes per layer as stored)."""
        cfg, it = self.cfg, self.dtype.itemsize
        out, ex = [0] * cfg.num_hidden_layers, [0] * cfg.num_hidden_layers
        for k in self.ck.idx:
            m = re.match(re.escape(PFX) + r"layers\.(\d+)\.", k)
            if not m or int(m.group(1)) >= cfg.num_hidden_layers:
                continue
            L = int(m.group(1))
            shp, dt = self.ck.shape(k)
            n = int(np.prod(shp)) if shp else 1
            if ".mlp.experts." in k:
                ex[L] += n * {"F8_E4M3": 1, "U8": 1, "BF16": 2, "F16": 2}.get(dt, 4)
            elif not k.endswith(("weight_scale_inv", "weight_scale", "weight_scale_2", "weight_global_scale",
                                 "input_scale", "input_global_scale")):
                out[L] += n * (2 if dt == "U8" else 1) * it
        return out, ex

    def expert_plan(self, L, dev):
        """[(device, [(b0, b1), ...])] shards of layer L's routed experts."""
        if not self.ep:
            return [(dev, [(b0, min(self.cfg.n_routed_experts, b0 + self.G))
                           for b0 in range(0, self.cfg.n_routed_experts, self.G)], False)]
        return [(self.devs[j], bl, j != self.place[L]) for j, bl in enumerate(self.ep_blocks) if bl]

    def selfcheck(self):
        """first block of every shard: block dequant == per-expert reference, bitwise (falls back to the torch
        dequant if the Triton path disagrees)."""
        def check():
            for lay in self.layers:
                for sh in lay.shards:
                    b0, b1 = sh.blocks[0]
                    gu, dn = sh.store.dequant(b0, b1)
                    for j, e in enumerate(range(b0, b1)):
                        rg, rd = sh.store.ref(e)
                        if not (torch.equal(gu[j], rg) and torch.equal(dn[j], rd)):
                            return f"L{lay.L} expert {e} on {sh.dev} ({sh.store.fmt})"
            return None
        bad = check()
        if bad and _DQ["triton"]:
            log(f"WARNING triton dequant mismatch at {bad}: falling back to torch dequant")
            _DQ["triton"] = False
            bad = check()
        assert bad is None, f"dequant self-check failed: {bad}"
        log(f"dequant self-check OK ({'triton' if _DQ['triton'] else 'torch'}; block dequant == reference, bitwise) "
            f"on every MoE layer / shard")

    # ---------------------------------------------------------------- passes
    def embed_tokens(self, tok):
        e = self.embed[tok.to(self.embed.device)]
        return e[:, None].expand(-1, self.cfg.hc_mult, -1).contiguous()

    def final_logits(self, H):
        H = H.to(self.head.device)
        z = F.linear(self.norm(self.hh(H[None]))[0], self.head)
        return z.float()

    def decode(self, tok, st):
        """tok [NS] long; st.pos_h = positions being fed -> logits [NS, V] fp32 (last device)."""
        st.push()
        H = self.embed_tokens(tok)
        for lay in self.layers:
            H = H.to(lay.dev)
            H = lay.forward_decode(H, st)
        return self.final_logits(H)

    def prefill(self, wv, all_logits=False):
        """-> logits of each sequence's last token [nseq, V] fp32 (or per-row top-64 logprobs if all_logits)."""
        H = self.embed_tokens(wv.tok)
        for lay in self.layers:
            H = H.to(lay.dev)
            H = lay.forward_prefill(H, wv)
        if not all_logits:
            return self.final_logits(H[torch.tensor(wv.last_rows(), device=H.device)])
        vs, is_ = [], []
        for c0 in range(0, wv.N, 2048):
            lp = F.log_softmax(self.final_logits(H[c0:c0 + 2048]), -1)
            v, i = lp.topk(64, -1)
            vs.append(v.half().cpu())
            is_.append(i.int().cpu())
        return torch.cat(vs), torch.cat(is_)

    def reset_slot(self, s):
        for lay in self.layers:
            lay.reset_slot(s)

    def records(self, s, n):
        """routing rows 0..n-1 of slot s -> {L: (ids uint16 [n,K], w fp16, xn fp32)} on the host."""
        out = {}
        for lay in self.layers:
            if lay.sparse:
                i, w, x = lay.records(s, n)
                out[lay.L] = (i.cpu().numpy().astype(np.uint16), w.cpu().numpy().copy(), x.cpu().numpy().copy())  # no alias
        return out


# ------------------------------------------------------------------ sampling
def sample(logits, temp, top_p, gen):
    """logits [n, V] fp32 -> tok [n], entropy [n] (nats, of the temp-scaled full distribution), lp of tok.
    HF top-p as gen.sample_top_p (keep the smallest prefix with cum prob >= top_p); temp 0 = greedy."""
    bad = ~torch.isfinite(logits).all(-1)
    if bad.any():
        logits = logits.masked_fill(bad[:, None], 0.0)
    if temp <= 0:
        lpa = F.log_softmax(logits, -1)
        tok = lpa.argmax(-1)
    else:
        lpa = F.log_softmax(logits / temp, -1)
        p = lpa.exp()
        sp, si = torch.sort(p, -1, descending=True)
        cs = sp.cumsum(-1)
        sp = sp.masked_fill((cs - sp) > top_p, 0)
        sp = sp / sp.sum(-1, keepdim=True)
        j = torch.multinomial(sp, 1, generator=gen)
        tok = si.gather(1, j)[:, 0]
        del sp, si, cs
    ent = -(lpa.exp() * lpa).nan_to_num().sum(-1)
    lp = lpa.gather(1, tok[:, None])[:, 0]
    return tok, ent, lp, bad


# ------------------------------------------------------------------ tasks
def load_tasks(a, tokzr):
    tasks = []
    if a.smoke:
        rows = [dict(id=f"smoke{i}", messages=[dict(role="user", content=p)]) for i, p in enumerate(SMOKE_PROMPTS)]
    elif a.mode == "tf":
        rows = json.load(open(a.tasks))
    else:
        rows = [json.loads(l) for l in open(a.input) if l.strip()]
    skipped = collections.Counter()
    for k, r in enumerate(rows):
        pid = str(r.get("id", k))
        if a.mode == "tf":
            t = np.asarray(r["tokens"], np.int64)
            tasks.append(dict(id=pid, group=pid, ids=t[: int(r["split"])], forced=t[int(r["split"]):], max_new=len(t) - int(r["split"])))
            continue
        if "prompt_ids" in r:
            ids = list(r["prompt_ids"])
        elif "messages" in r:
            msgs = r["messages"]
            if any(isinstance(m.get("content"), list) and any(c.get("type") in ("image", "image_url", "video")
                                                              for c in m["content"] if isinstance(c, dict))
                   for m in msgs):
                skipped["image"] += 1
                continue
            kw = {}
            eff = r.get("reasoning_effort", a.effort)
            if eff:
                kw["reasoning_effort"] = eff
            enc = tokzr.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, **kw)
            ids = list(enc["input_ids"] if hasattr(enc, "keys") else enc)
        elif "text" in r:
            ids = tokzr.encode(r["text"], add_special_tokens=False)
        else:
            skipped["no prompt"] += 1
            continue
        mn = int(r.get("max_new", a.max_new))
        for j in range(a.n_samples):
            sid = pid if a.n_samples == 1 else f"{pid}#s{j}"
            tasks.append(dict(id=sid, group=str(r.get("group", pid)), ids=np.asarray(ids, np.int64), max_new=mn,
                              src=r.get("src"), cat=r.get("cat")))
    if skipped:
        log(f"skipped prompts: {dict(skipped)}")
    return tasks


# ------------------------------------------------------------------ output
class Writer:
    """shards of finished sequences (PRIVATE).  Each shard = one jF 'rank' after finalize()."""

    def __init__(self, out, moe_layers, shard_rows, meta):
        self.out, self.layers, self.shard_rows, self.meta = out, moe_layers, shard_rows, meta
        os.makedirs(f"{out}/shards", exist_ok=True)
        self.k = len(sorted(glob.glob(f"{out}/shards/s?????")))
        self._new()

    def _new(self):
        self.seqs, self.tok, self.ent, self.lp, self.gen, self.rec, self.rows = [], [], [], [], [], {L: [] for L in self.layers}, 0

    def done_ids(self):
        ids = set()
        for f in glob.glob(f"{self.out}/shards/s?????/seqs.json"):
            ids |= {q["id"] for q in json.load(open(f))["seqs"]}
        return ids

    def add(self, meta, tok, rec, ent, lp, gen):
        n = len(tok)
        self.seqs.append(dict(meta, rows=n))
        self.tok.append(tok.astype(np.int32))
        self.ent.append(ent.astype(np.float32))
        self.lp.append(lp.astype(np.float32))
        self.gen.append(gen)
        for L in self.layers:
            self.rec[L].append(rec[L])
        self.rows += n
        if self.rows >= self.shard_rows:
            self.flush()

    def flush(self):
        if not self.seqs:
            return
        d = f"{self.out}/shards/s{self.k:05d}"
        tmp = d + ".part"
        os.makedirs(tmp, exist_ok=True)
        for L in self.layers:
            r = self.rec[L]
            np.savez(f"{tmp}/L{L}.npz", ids=np.concatenate([x[0] for x in r]), w=np.concatenate([x[1] for x in r]),
                     xn=np.concatenate([x[2] for x in r]))
        np.save(f"{tmp}/tok.npy", np.concatenate(self.tok))
        np.savez(f"{tmp}/aux.npz", ent=np.concatenate(self.ent), lp=np.concatenate(self.lp))
        json.dump(dict(N=self.rows, seqs=self.seqs, decode=self.meta), open(f"{tmp}/seqs.json", "w"))
        with open(f"{tmp}/gen.jsonl", "w") as f:
            for g in self.gen:
                f.write(json.dumps(g) + "\n")
        os.replace(tmp, d)
        with open(f"{self.out}/gen.jsonl", "a") as f:
            for g in self.gen:
                f.write(json.dumps(g) + "\n")
        log(f"wrote shard s{self.k:05d}: {len(self.seqs)} seqs {self.rows} rows")
        self.k += 1
        self._new()
        finalize(self.out)


def finalize(out):
    """(re)link complete shards as ranks r of W in OUT (jf/blocks37.py decode layout) + index.json."""
    sh = sorted(d for d in glob.glob(f"{out}/shards/s?????") if os.path.exists(f"{d}/seqs.json"))
    for f in glob.glob(f"{out}/*.r*of*.*"):
        if os.path.islink(f):
            os.remove(f)
    W = len(sh)
    tasks, meta = [], None
    for r, d in enumerate(sh):
        rel = os.path.relpath(d, out)
        for f in glob.glob(f"{d}/L*.npz"):
            L = os.path.basename(f)[1:-4]
            os.symlink(f"{rel}/L{L}.npz", f"{out}/L{L}.r{r}of{W}.npz")
        os.symlink(f"{rel}/tok.npy", f"{out}/tok.r{r}of{W}.npy")
        os.symlink(f"{rel}/aux.npz", f"{out}/aux.r{r}of{W}.npz")
        os.symlink(f"{rel}/seqs.json", f"{out}/seqs.r{r}of{W}.json")
        j = json.load(open(f"{d}/seqs.json"))
        meta = j.get("decode", meta)
        o = 0
        for q in j["seqs"]:
            tasks.append(dict(id=q["id"], g=r, off=o, end=o + q["rows"], prompt_len=q["prompt_len"],
                              n_dec=q["n_gen"], stopped=q["stopped"]))
            o += q["rows"]
    json.dump(dict(source="dec37", W=W, tasks=tasks, decode=meta,
                   note="rank r = shard r; rows per task: prompt 0..prompt_len-1 + decode prompt_len..prompt_len+n_dec-2 "
                        "(the n_dec-th sampled token is the stop / cap token and is not fed)"),
              open(f"{out}/index.json.part", "w"))
    os.replace(f"{out}/index.json.part", f"{out}/index.json")
    return W


# ------------------------------------------------------------------ driver
STOP = {"flag": False}


def run(a):
    os.makedirs(a.out, exist_ok=True)
    u, r = check_host(a.host_limit_gb, a.rss_limit_gb, "at start")
    log(f"host unreclaimable {u:.0f} GB (limit {a.host_limit_gb:.0f}), own RSS {r:.1f} GB")
    torch.set_num_threads(a.threads)
    _TF32["all"] = a.tf32 == "all"
    _TF32["attn"] = a.tf32 in ("all", "attn")
    torch.backends.cuda.matmul.allow_tf32 = _TF32["all"]
    torch.backends.cudnn.allow_tf32 = _TF32["all"]
    from transformers import AutoTokenizer
    tokzr = AutoTokenizer.from_pretrained(a.ckpt)
    tasks = load_tasks(a, tokzr)
    moe_layers = None
    meta = dict(source="dec37", model=a.ckpt, mode=a.mode, temp=a.temp, top_p=a.top_p, stop_ids=list(STOP_IDS),
                effort=a.effort, seed=a.seed, tf32=a.tf32, dtype=a.dtype, started=time.strftime("%Y-%m-%d %H:%M:%S %Z"))
    if a.mode == "gen":
        w0 = Writer.__new__(Writer)
        w0.out = a.out
        done = w0.done_ids() if os.path.isdir(f"{a.out}/shards") else set()
        if done:
            n0 = len(tasks)
            tasks = [t for t in tasks if t["id"] not in done]
            log(f"resume: {n0 - len(tasks)} done, {len(tasks)} to go")
    maxP = max((len(t["ids"]) for t in tasks), default=1)
    Smax = a.smax or (min(maxP, a.max_prompt) + max(t["max_new"] for t in tasks) + 1 if tasks else 64)
    keep = [t for t in tasks if len(t["ids"]) + 1 < Smax and len(t["ids"]) <= a.max_prompt]
    if len(keep) < len(tasks):
        log(f"dropped {len(tasks) - len(keep)} prompts longer than max_prompt {a.max_prompt} / Smax {Smax}")
    tasks = keep
    if not tasks:
        log("nothing to do")
        return
    NS = min(a.slots, len(tasks)) if a.mode == "tf" or a.slots > len(tasks) else a.slots
    eng = Engine(a, NS, Smax)
    moe_layers = [lay.L for lay in eng.layers if lay.sparse]
    wr = Writer(a.out, moe_layers, a.shard_rows, meta)
    dl = eng.dev_of[-1]
    gen = torch.Generator(device=dl)
    gen.manual_seed(a.seed + 1000003 * wr.k)
    st = StepState(NS, sorted(set(eng.dev_of), key=str))
    pending = collections.deque(tasks)
    slots = [None] * NS
    cur = torch.zeros(NS, dtype=torch.long)
    tf_out = f"{a.out}/tf"
    if a.mode == "tf":
        os.makedirs(tf_out, exist_ok=True)
    t_start, n_tok, n_step, last_log = time.time(), 0, 0, time.time()
    n_pre, t_pre = 0, 0.0

    def finish(s, why):
        q = slots[s]
        P, G = q["P"], len(q["gen"])
        rows = P + G - 1
        if why != "nan":
            rec = eng.records(s, rows)
            tok = np.concatenate([q["ids"], np.asarray(q["gen"][:-1], np.int64)]).astype(np.int32)
            ent = np.full(rows, np.nan, np.float32)
            lp = np.full(rows, np.nan, np.float32)
            ent[P - 1:P - 1 + len(q["ent"])] = q["ent"][: rows - P + 1]
            lp[P - 1:P - 1 + len(q["lp"])] = q["lp"][: rows - P + 1]
            gi = q["gen"]
            te = gi.index(THINK_END) if THINK_END in gi else -1
            m = dict(id=q["id"], prompt_len=P, group=q["group"], n_gen=G, stopped=why == "stop",
                     stop_id=int(gi[-1]) if why == "stop" else -1, think_end=te)
            g = dict(m, gen=[int(x) for x in gi], src=q.get("src"), cat=q.get("cat"),
                     text=tokzr.decode(gi, skip_special_tokens=False) if not a.no_text else None)
            wr.add(m, tok, rec, ent, lp, g)
            if a.mode == "tf":
                v = np.concatenate([q["pre_v"].numpy(), np.stack(q["tv"])]) if q["tv"] else q["pre_v"].numpy()
                i = np.concatenate([q["pre_i"].numpy(), np.stack(q["ti"])]) if q["ti"] else q["pre_i"].numpy()
                np.savez(f"{tf_out}/{q['id']}.npz", v=v, i=i, P=P)
        else:
            log(f"WARNING {q['id']}: non-finite logits at gen {G}; sequence dropped")
        eng.reset_slot(s)
        slots[s] = None
        st.pos_h[s] = 0
        cur[s] = 0

    def on_sig(*_):
        STOP["flag"] = True
        log("signal: stopping after this step (finished sequences are flushed; in-flight ones are dropped)")

    signal.signal(signal.SIGTERM, on_sig)
    signal.signal(signal.SIGINT, on_sig)
    while True:
        if STOP["flag"] or os.path.exists(f"{a.out}/STOP"):
            break
        free = [s for s in range(NS) if slots[s] is None]
        active = NS - len(free)
        if pending and free and (len(free) >= max(1, int(NS * a.refill_frac)) or active == 0 or a.mode == "tf"):
            take, toks, budget = [], [], a.wave_tokens
            while pending and len(take) < len(free):
                t = pending[0]
                if toks and budget - len(t["ids"]) < 0:
                    break
                budget -= len(t["ids"])
                take.append(pending.popleft())
                toks.append(t["ids"])
            sl = free[: len(take)]
            t0 = time.time()
            wv = Wave(sl, toks)
            if a.mode == "tf":
                pv, pi = eng.prefill(wv, all_logits=True)
                last = None
            else:
                last = eng.prefill(wv)
            if a.mode == "tf":
                for j, (s, t) in enumerate(zip(sl, take)):
                    o, T = wv.off[j], wv.T[j]
                    slots[s] = dict(t, P=T, gen=[], ent=[], lp=[], pre_v=pv[o:o + T], pre_i=pi[o:o + T], tv=[], ti=[])
                    if len(t["forced"]) == 0:
                        slots[s]["gen"] = [0]
                        finish(s, "cap")
                        continue
                    nt = int(t["forced"][0])
                    slots[s]["gen"].append(nt)
                    slots[s]["ent"].append(np.nan)
                    slots[s]["lp"].append(np.nan)
                    st.pos_h[s] = T
                    cur[s] = nt
            else:
                tk, en, lp, bad = sample(last, a.temp, a.top_p, gen)
                tk, en, lp, bad = tk.cpu().numpy(), en.cpu().numpy(), lp.cpu().numpy(), bad.cpu().numpy()
                for j, (s, t) in enumerate(zip(sl, take)):
                    slots[s] = dict(t, P=wv.T[j], gen=[int(tk[j])], ent=[float(en[j])], lp=[float(lp[j])])
                    st.pos_h[s] = wv.T[j]
                    cur[s] = int(tk[j])
                    if bad[j]:
                        finish(s, "nan")
                    elif int(tk[j]) in STOP_IDS:
                        finish(s, "stop")
                    elif t["max_new"] <= 1 or wv.T[j] + 1 >= Smax:
                        finish(s, "cap")
            n_pre += wv.N
            t_pre += time.time() - t0
            log(f"prefill wave: {len(take)} seqs {wv.N} tokens {time.time() - t0:.1f}s; pending {len(pending)}")
            continue
        if active == 0:
            if not pending:
                break
            continue
        logits = eng.decode(cur, st)
        n_step += 1
        act = [s for s in range(NS) if slots[s] is not None]
        if a.mode == "tf":
            lpa = F.log_softmax(logits, -1)
            v, i = lpa.topk(64, -1)
            v, i = v.half().cpu().numpy(), i.int().cpu().numpy()
            for s in act:
                q = slots[s]
                q["tv"].append(v[s])
                q["ti"].append(i[s])
                k = len(q["gen"])
                st.pos_h[s] += 1
                if k >= len(q["forced"]):
                    q["gen"].append(0)
                    finish(s, "cap")
                    continue
                nt = int(q["forced"][k])
                q["gen"].append(nt)
                q["ent"].append(np.nan)
                q["lp"].append(np.nan)
                cur[s] = nt
            n_tok += len(act)
        else:
            logits[[s for s in range(NS) if slots[s] is None]] = 0.0
            tk, en, lp, bad = sample(logits, a.temp, a.top_p, gen)
            tk, en, lp, bad = tk.cpu().numpy(), en.cpu().numpy(), lp.cpu().numpy(), bad.cpu().numpy()
            for s in act:
                q = slots[s]
                q["gen"].append(int(tk[s]))
                q["ent"].append(float(en[s]))
                q["lp"].append(float(lp[s]))
                st.pos_h[s] += 1
                cur[s] = int(tk[s])
                G = len(q["gen"])
                if bad[s]:
                    finish(s, "nan")
                elif int(tk[s]) in STOP_IDS:
                    finish(s, "stop")
                elif G >= q["max_new"] or q["P"] + G >= Smax:
                    finish(s, "cap")
            n_tok += len(act)
        if time.time() - last_log > a.log_every:
            el = time.time() - t_start
            u, r = check_host(a.host_limit_gb, a.rss_limit_gb, f"at step {n_step}")
            mem = " ".join(f"{torch.cuda.max_memory_allocated(d) / 1e9:.0f}" for d in st.devs if d.type == "cuda")
            log(f"step {n_step}: active {len(act)}/{NS} pending {len(pending)} | decode tok {n_tok} "
                f"({n_tok / el:.1f} tok/s overall) prefill {n_pre} tok {t_pre:.0f}s | done {wr.k} shards + "
                f"{len(wr.seqs)} buffered | host {u:.0f} GB rss {r:.1f} | peak GB {mem}")
            last_log = time.time()
    wr.flush()
    W = finalize(a.out)
    el = time.time() - t_start
    log(f"finished: {n_step} steps, {n_tok} decode tokens in {el:.0f}s ({n_tok / max(el, 1e-9):.1f} tok/s), "
        f"prefill {n_pre} tokens; {W} shards in {a.out}")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=CK_DEFAULT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--mode", choices=["gen", "tf"], default="gen")
    ap.add_argument("--input", help="jsonl: {id, messages | prompt_ids | text, [reasoning_effort], [max_new], [group]}")
    ap.add_argument("--tasks", help="tf mode: json [{id, tokens, split}]")
    ap.add_argument("--smoke", action="store_true", help="built-in smoke prompts")
    ap.add_argument("--finalize", metavar="OUT", help="only (re)build the jF symlink layout of OUT")
    ap.add_argument("--devices", default=",".join(f"cuda:{i}" for i in range(8)))
    ap.add_argument("--place", default="auto", help="auto | comma list of device index per layer")
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--slots", type=int, default=256, help="NS concurrent sequences (= decode batch)")
    ap.add_argument("--max-new", type=int, default=8192)
    ap.add_argument("--max-prompt", type=int, default=32768)
    ap.add_argument("--smax", type=int, default=0, help="per-slot capacity (prompt + generation); 0 = auto")
    ap.add_argument("--n-samples", type=int, default=1)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--effort", default=None, help="chat template reasoning_effort (template default: max)")
    ap.add_argument("--seed", type=int, default=20261006)
    ap.add_argument("--tf32", choices=["all", "attn", "none"], default="all",
                    help="all = capture37g parity (TF32 matmuls incl router / KDA chunk); KDA decode step always exact fp32")
    ap.add_argument("--expert-block", type=int, default=0,
                    help="experts per dequant block; 0 = auto (16, or ceil(288/ndev) = 36 with --ep)")
    ap.add_argument("--ep", action="store_true",
                    help="expert parallel: every layer's routed experts sharded by block over all devices (each "
                         "expert dequantised once per step, all GPUs busy in every MoE layer); bitwise = non-ep for "
                         "the same --expert-block. Default off until GPU-verified")
    ap.add_argument("--dq", choices=["auto", "torch", "triton"], default="torch",
                    help="fp8 expert dequant: torch (2-pass) or fused one-pass Triton (bitwise equal, self-checked "
                         "at start with fallback); auto = triton on CUDA when importable. Default torch until GPU-verified")
    ap.add_argument("--tok-chunk", type=int, default=4096)
    ap.add_argument("--kda-piece", type=int, default=256)
    ap.add_argument("--kda-tokens", type=int, default=512, help="KDA prefill batch x piece token budget")
    ap.add_argument("--attn-budget-gb", type=float, default=1.5)
    ap.add_argument("--wave-tokens", type=int, default=32768)
    ap.add_argument("--refill-frac", type=float, default=0.125)
    ap.add_argument("--shard-rows", type=int, default=262144)
    ap.add_argument("--reserve-gb", type=float, default=8.0)
    ap.add_argument("--host-limit-gb", type=float, default=HOST_LIMIT_GB)
    ap.add_argument("--rss-limit-gb", type=float, default=64.0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--log-every", type=float, default=60.0)
    ap.add_argument("--no-text", action="store_true")
    return ap.parse_args(argv)


def main():
    a = parse_args()
    if a.finalize:
        print("W =", finalize(a.finalize))
        return
    assert a.smoke or a.input or (a.mode == "tf" and a.tasks), "need --input, --smoke or --mode tf --tasks"
    assert a.kda_piece % 64 == 0
    run(a)


if __name__ == "__main__":
    main()
