"""T34: decoder for davidsyoung/GLM-5.3-EXL3-TR3-3.42bpw (rev 2f404fc9) routed experts -> dense weights.

Layout (verified on the shards, see REPORT):
  model.layers.{L}.mlp.experts.{E}.{proj}.rank{r}.{trellis,suh,svh,mcg}, r = 0..3 (TP4 pre-sliced)
    gate/up rank r = output rows [512r, 512r+512): EXL3 matrix k=in=6144, n=out=512   trellis [384, 32, 16K]
    down    rank r = input cols  [512r, 512r+512): EXL3 matrix k=in=512,  n=out=6144  trellis [32, 384, 16K]
  K (3 or 4) per (layer, expert) = trellis.shape[-1] / 16 (== tier_bitmap.json); mcg = int32 0xCBAC1FED.
Each rank tensor is a plain exllamav3 LinearEXL3 (mcg codebook).  Decode == LinearEXL3.get_weight_tensor:
  Q (fp16, rotated) = mcg_lut[state], state_i = 16 bits ending at bit (i+1)K of the tile's tail-biting bitstream
  (pack_trellis: K bits per weight MSB-first in uint16 words, uint32 halves swapped = little-endian uint32 read
  gives the big-endian 32-bit stream), encoder order -> row-major via tensor_core_perm_i;
  W^T = ((had128_l(Q) * suh[:,None]) had128_r) * svh[None,:], fp32 matmuls rounded to fp16 between steps.
Pure torch; runs on CPU or GPU.  Output [out, in] (HF layout), fp16 values (exactly representable in fp32).
"""
import json
import math
import os
from functools import lru_cache

import torch

MCG_MULT = 0xCBAC1FED
PROJ = ("gate_proj", "up_proj", "down_proj")
NRANK = 4


@lru_cache
def mcg_lut(device="cpu"):
    """exllamav3 codebook 1 (mcg): x = s*0xCBAC1FED; x = (x & 0x8fff8fff) ^ 0x3b603b60; fp16(hi) + fp16(lo)."""
    s = torch.arange(65536, dtype=torch.int64)
    x = (s * MCG_MULT) % 2**32
    x = (x & 0x8fff8fff) ^ 0x3b603b60
    lo = (x & 0xFFFF).to(torch.int32).to(torch.int16).view(torch.float16).float()
    hi = (x >> 16).to(torch.int32).to(torch.int16).view(torch.float16).float()
    return (lo + hi).half().to(device)          # __hadd: exact sum, one rounding


@lru_cache
def _perm_i(device="cpu"):
    perm = [0] * 256
    for t in range(32):
        r0 = (t % 4) * 2; r1 = r0 + 1; r2 = r0 + 8; r3 = r0 + 9
        c0 = t // 4; c1 = c0 + 8
        perm[t * 8:t * 8 + 8] = [r0 * 16 + c0, r1 * 16 + c0, r2 * 16 + c0, r3 * 16 + c0,
                                 r0 * 16 + c1, r1 * 16 + c1, r2 * 16 + c1, r3 * 16 + c1]
    return torch.argsort(torch.tensor(perm)).to(device)


@lru_cache
def _state_index(K, device="cpu"):
    """(a, a1, shift): state_i = ((w32[a] << 32 | w32[a1]) >> shift) & 0xFFFF, bit p = ((i+1)K - 16) mod 256K."""
    nbits = 256 * K
    i = torch.arange(256, dtype=torch.int64)
    p = ((i + 1) * K - 16) % nbits
    a = p // 32
    o = p % 32
    nw = nbits // 32
    return a.to(device), ((a + 1) % nw).to(device), (48 - o).to(device)


@lru_cache
def _had128(device="cpu"):
    # exllamav3 get_hadamard(128): no data file for 128 -> Sylvester doubling from hadamard_1.txt ("+")
    h = torch.ones(1, 1)
    while h.shape[0] < 128:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return (h * (1 / math.sqrt(128))).to(device)


def unpack_states(trellis):
    """trellis int16 [kt, nt, 16K] -> uint16 states as int64 [kt, nt, 256] (encoder / tensor-core order)."""
    kt, nt, w = trellis.shape
    K = w // 16
    u = trellis.reshape(-1, w).to(torch.int32) & 0xFFFF
    w32 = ((u[:, 1::2].to(torch.int64) << 16) | u[:, 0::2].to(torch.int64))      # [T, 8K] big-endian stream words
    a, a1, sh = _state_index(K, trellis.device)
    v = ((w32[:, a] << 32) | w32[:, a1]) >> sh
    return (v & 0xFFFF).view(kt, nt, 256)


def decode_inner(trellis):
    """Rotated-basis fp16 weight [k, n] (== ext.reconstruct)."""
    kt, nt, _ = trellis.shape
    st = unpack_states(trellis)
    q = mcg_lut(str(trellis.device))[st.view(-1, 256)][:, _perm_i(trellis.device)]    # row-major tiles
    return q.view(kt, nt, 16, 16).permute(0, 2, 1, 3).reshape(kt * 16, nt * 16)


HAD_DTYPE = torch.float64     # exllamav3 uses fp32 cuBLAS; fp64 makes the fp16 result device-independent (CPU == GPU)


def decode_linear(trellis, suh, svh, had_dtype=None):
    """LinearEXL3.get_weight_tensor: [k=in, n=out] fp16."""
    dt = had_dtype or HAD_DTYPE
    w = decode_inner(trellis)
    k, n = w.shape
    had = _had128(str(w.device)).to(dt)
    w = (had @ w.to(dt).view(-1, 128, n)).view(k, n).half()
    w *= suh.half().unsqueeze(1)
    w = (w.to(dt).view(k, -1, 128) @ had).view(k, n).half()
    w *= svh.half().unsqueeze(0)
    return w


class TR3Layer:
    """Header-indexed reader for one model-layer-{L:03d}.safetensors (pread per tensor, no mmap of 4.5 GB)."""

    def __init__(self, root, layer):
        self.f = f"{root}/model-layer-{layer:03d}.safetensors"
        with open(self.f, "rb") as fh:
            n = int.from_bytes(fh.read(8), "little")
            self.hdr = json.loads(fh.read(n))
        self.base = 8 + n
        self.layer = layer
        self.hdr.pop("__metadata__", None)

    _DT = {"I16": torch.int16, "F16": torch.float16, "I32": torch.int32, "BF16": torch.bfloat16,
           "F32": torch.float32}

    def get(self, key, device="cpu"):
        h = self.hdr[key]
        a, b = h["data_offsets"]
        with open(self.f, "rb") as fh:
            fh.seek(self.base + a)
            buf = bytearray(fh.read(b - a))
        t = torch.frombuffer(buf, dtype=self._DT[h["dtype"]]).reshape(h["shape"]) if b > a else \
            torch.empty(h["shape"], dtype=self._DT[h["dtype"]])
        return t.to(device)

    def K(self, e):
        return self.hdr[f"model.layers.{self.layer}.mlp.experts.{e}.gate_proj.rank0.trellis"]["shape"][-1] // 16

    def raw(self, e, device="cpu"):
        p = f"model.layers.{self.layer}.mlp.experts.{e}"
        return {pr: [{k: self.get(f"{p}.{pr}.rank{r}.{k}", device) for k in ("trellis", "suh", "svh", "mcg")}
                     for r in range(NRANK)] for pr in PROJ}

    def expert(self, e, device="cpu", raw=None, had_dtype=None):
        """{'gate_proj': [2048, 6144], 'up_proj': [2048, 6144], 'down_proj': [6144, 2048]} fp16, [out, in]."""
        raw = raw or self.raw(e, device)
        out = {}
        for pr in PROJ:
            parts = []
            for r in raw[pr]:
                assert int(r["mcg"].view(torch.int32).item()) & 0xFFFFFFFF == MCG_MULT, "non-mcg codebook"
                parts.append(decode_linear(r["trellis"], r["suh"], r["svh"], had_dtype).T)   # [out_slice, in] or [out, in_slice]
            out[pr] = torch.cat(parts, 0 if pr != "down_proj" else 1).contiguous()
        return out


def bits_of_expert(layer_reader, e):
    """Exact storage bits: trellis 256*K per 16x16 tile + fp16 suh/svh per rank tensor (mcg word excluded)."""
    b = 0
    p = f"model.layers.{layer_reader.layer}.mlp.experts.{e}"
    for pr in PROJ:
        for r in range(NRANK):
            for k in ("trellis", "suh", "svh"):
                a, c = layer_reader.hdr[f"{p}.{pr}.rank{r}.{k}"]["data_offsets"]
                b += 8 * (c - a)
    return b
