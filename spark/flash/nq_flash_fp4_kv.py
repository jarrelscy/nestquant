"""Experimental, opt-in NoPE sparse MLA cache for Flash (TP1, eager only).

Each token stores 512 E2M1 values, plus 32 little-endian FP16 group scales.
The cache uses 320 bytes per token, including scales. This is *not* the NVFP4
FlashInfer layout. KDA recurrent state and DSA indexer caches stay unchanged.
"""

import math

import torch
import triton as tr
import triton.language as tl

LATENT = 512
GROUP = 16
BYTES = 320
FORMAT = "nq_flash_fp4_g16_v1"


@tr.jit
def _unpack(v):
    m = v & 7
    a = tl.where(
        m < 2,
        m * 0.5,
        tl.where(
            m < 4,
            1.0 + (m - 2) * 0.5,
            tl.where(m < 6, 2.0 + (m - 4), 4.0 + (m - 6) * 2.0),
        ),
    )
    return tl.where((v & 8) != 0, -a, a)


@tr.jit
def _pack(
    X,
    C,
    S,
    XS: tl.constexpr,
    SS: tl.constexpr,
    PAGE: tl.constexpr,
    CS0: tl.constexpr,
    CS1: tl.constexpr,
    NSLOTS: tl.constexpr,
):
    t = tl.program_id(0)
    g = tl.arange(0, 32)
    j = tl.arange(0, 16)
    slot = tl.load(S + t * SS).to(tl.int64)
    valid = (slot >= 0) & (slot < NSLOTS)
    tl.device_assert((slot < 0) | (slot < NSLOTS), "FP4 KV slot out of bounds")
    x = tl.load(X + t * XS + g[:, None] * 16 + j[None, :]).to(tl.float32)
    # Use the *stored* rounded scale when choosing codes. Nonfinite input is
    # invalid, as for the surrounding model; debug builds diagnose it.
    tl.device_assert((x == x) & (tl.abs(x) <= 393024.0), "FP4 KV input out of range")
    sf = tl.minimum(
        tl.maximum(tl.max(tl.abs(x), 1) / 6.0, 5.960464477539063e-8), 65504.0
    ).to(tl.float16)
    y = tl.abs(x) / sf[:, None].to(tl.float32)
    # E2M1 round-to-nearest-even, including all seven bin boundaries.
    z = tl.where(
        y <= 0.25,
        0,
        tl.where(
            y < 0.75,
            1,
            tl.where(
                y <= 1.25,
                2,
                tl.where(
                    y < 1.75,
                    3,
                    tl.where(
                        y <= 2.5, 4, tl.where(y < 3.5, 5, tl.where(y <= 5.0, 6, 7))
                    ),
                ),
            ),
        ),
    )
    z = (z | tl.where(x < 0, 8, 0)).to(tl.uint8)
    even = tl.gather(z, tl.broadcast_to(tl.arange(0, 8)[None, :] * 2, (32, 8)), 1)
    odd = tl.gather(z, tl.broadcast_to(tl.arange(0, 8)[None, :] * 2 + 1, (32, 8)), 1)
    base = slot // PAGE * CS0 + slot % PAGE * CS1
    tl.store(
        C + base + g[:, None] * 8 + tl.arange(0, 8)[None, :], even | (odd << 4), valid
    )
    bits = sf.to(tl.uint16, bitcast=True)
    tl.store(C + base + 256 + g * 2, (bits & 255).to(tl.uint8), valid)
    tl.store(C + base + 257 + g * 2, (bits >> 8).to(tl.uint8), valid)


@tr.jit
def _attention(
    Q,
    C,
    IDX,
    OUT,
    L,
    H: tl.constexpr,
    K: tl.constexpr,
    QS: tl.constexpr,
    QH: tl.constexpr,
    IS: tl.constexpr,
    PAGE: tl.constexpr,
    CS0: tl.constexpr,
    CS1: tl.constexpr,
    NSLOTS: tl.constexpr,
    SCALE: tl.constexpr,
    SPLITS: tl.constexpr,
    BK: tl.constexpr = 32,
    BH: tl.constexpr = 16,
):
    t = tl.program_id(0)
    hg = tl.program_id(1)
    sp = tl.program_id(2)
    h = hg * BH + tl.arange(0, BH)
    d = tl.arange(0, 512)
    q = tl.load(Q + t * QS + h[:, None] * QH + d[None, :], h[:, None] < H, 0)
    m = tl.full((BH,), -float("inf"), tl.float32)
    den = tl.zeros((BH,), tl.float32)
    acc = tl.zeros((BH, 512), tl.float32)
    for b in range(tr.cdiv(K, SPLITS * BK)):
        k = (sp * tr.cdiv(K, SPLITS * BK) + b) * BK + tl.arange(0, BK)
        slot = tl.load(IDX + t * IS + k, k < K, -1).to(tl.int64)
        valid = (slot >= 0) & (slot < NSLOTS)
        tl.device_assert((slot < 0) | (slot < NSLOTS), "FP4 KV index out of bounds")
        base = slot // PAGE * CS0 + slot % PAGE * CS1
        packed = tl.load(C + base[:, None] + d[None, :] // 2, valid[:, None], 0)
        nib = (packed >> ((d[None, :] % 2) * 4)) & 15
        lo = tl.load(
            C + base[:, None] + 256 + (d[None, :] // 16) * 2, valid[:, None], 0
        ).to(tl.uint16)
        hi = tl.load(
            C + base[:, None] + 257 + (d[None, :] // 16) * 2, valid[:, None], 0
        ).to(tl.uint16)
        sf = (lo | (hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        v = (_unpack(nib) * sf).to(q.dtype)
        logits = tl.dot(q, tl.trans(v)).to(tl.float32) * SCALE
        logits = tl.where(valid[None, :], logits, -float("inf"))
        new_m = tl.maximum(m, tl.max(logits, 1))
        safe_m = tl.where(new_m == -float("inf"), 0.0, new_m)
        alpha = tl.exp(m - safe_m)
        p = tl.exp(logits - safe_m[:, None])
        acc = acc * alpha[:, None] + tl.dot(p.to(q.dtype), v)
        den = den * alpha + tl.sum(p, 1)
        m = new_m
    value = acc / tl.maximum(den[:, None], 1e-30)
    tl.store(
        OUT + ((t * SPLITS + sp) * H + h[:, None]) * 512 + d[None, :],
        value,
        h[:, None] < H,
    )
    tl.store(
        L + (t * SPLITS + sp) * H + h,
        tl.where(den > 0, m + tl.log(tl.maximum(den, 1e-30)), -float("inf")),
        h < H,
    )


@tr.jit
def _merge(
    P, L, OUT, H: tl.constexpr, SPLITS: tl.constexpr, OS: tl.constexpr, OH: tl.constexpr
):
    t = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.arange(0, SPLITS)
    d = tl.arange(0, 512)
    lognorm = tl.load(L + (t * SPLITS + s) * H + h)
    m = tl.max(lognorm, 0)
    m = tl.where(m == -float("inf"), 0.0, m)
    w = tl.exp(lognorm - m)
    w = w / tl.maximum(tl.sum(w, 0), 1e-30)
    p = tl.load(P + ((t * SPLITS + s[:, None]) * H + h) * 512 + d[None, :])
    tl.store(OUT + t * OS + h * OH + d, tl.sum(p * w[:, None], 0))


def _cache_bytes(cache):
    if cache.ndim != 3 or cache.shape[-1] != BYTES or cache.element_size() != 1:
        raise ValueError(
            "FP4 MLA cache must have shape [blocks, tokens, 320] and byte elements"
        )
    if cache.stride(-1) != 1 or cache.stride(1) < BYTES:
        raise ValueError("FP4 MLA requires contiguous bytes within a token")
    if cache.stride(0) < cache.shape[1] * cache.stride(1):
        raise ValueError("FP4 MLA pages must not overlap")
    if cache.shape[1] <= 0:
        raise ValueError("FP4 MLA page must contain tokens")
    return cache.view(torch.uint8)


def _check_device(cache, *tensors):
    if not cache.is_cuda or any(x.device != cache.device for x in tensors):
        raise ValueError("FP4 MLA operands must share one CUDA device")
    if torch.cuda.is_current_stream_capturing():
        raise ValueError(
            "FP4 MLA cache has not been validated with CUDA graphs; use eager"
        )


def pack(x, cache, slots):
    cache = _cache_bytes(cache)
    if x.ndim != 2 or x.shape[1] != LATENT or x.stride(1) != 1:
        raise ValueError("FP4 MLA requires contiguous 512-element latent rows")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("FP4 MLA input must be a floating tensor")
    if (
        slots.ndim != 1
        or slots.numel() != len(x)
        or slots.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("FP4 MLA slot mapping must have one integer per token")
    _check_device(cache, x, slots)
    if len(x):
        _pack[(len(x),)](
            x,
            cache,
            slots,
            x.stride(0),
            slots.stride(0),
            cache.shape[1],
            cache.stride(0),
            cache.stride(1),
            cache.shape[0] * cache.shape[1],
            num_warps=4,
        )


def attention(q, cache, indices, scale):
    cache = _cache_bytes(cache)
    if q.ndim != 3 or q.shape[-1] != LATENT or q.stride(-1) != 1 or q.shape[1] <= 0:
        raise ValueError("FP4 MLA query must have shape [tokens, heads, 512]")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("FP4 MLA query must be FP16 or BF16")
    if (
        indices.ndim != 2
        or indices.shape[0] != q.shape[0]
        or indices.stride(1) != 1
        or indices.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError(
            "FP4 MLA indices must have contiguous integer rows, one per query"
        )
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("FP4 MLA attention scale must be finite and positive")
    _check_device(cache, q, indices)
    n, h, _ = q.shape
    out = torch.empty_like(q)
    splits = 8
    # Scratch bounded independently of context and prefill chunk length. Query
    # heads and values retain native FP16/BF16; MMA accumulates in FP32.
    partial = torch.empty(
        (min(n, 16), splits, h, LATENT), device=q.device, dtype=torch.float32
    )
    lse = torch.empty((min(n, 16), splits, h), device=q.device, dtype=torch.float32)
    for start in range(0, n, 16):
        qs = q[start : start + 16]
        ix = indices[start : start + 16]
        nt = len(qs)
        _attention[(nt, tr.cdiv(h, 16), splits)](
            qs,
            cache,
            ix,
            partial,
            lse,
            h,
            ix.shape[1],
            qs.stride(0),
            qs.stride(1),
            ix.stride(0),
            cache.shape[1],
            cache.stride(0),
            cache.stride(1),
            cache.shape[0] * cache.shape[1],
            scale,
            splits,
            num_warps=8,
        )
        _merge[(nt, h)](
            partial,
            lse,
            out[start : start + nt],
            h,
            splits,
            out.stride(0),
            out.stride(1),
            num_warps=4,
        )
    return out


def install():
    """Install before construction, after the existing NoPE adapter.

    The vLLM CLI still uses ``--kv-cache-dtype fp8`` so indexer kernels retain
    their established FP8 format. Only explicitly stamped MLA specs use FP4.
    """
    from dataclasses import replace
    from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
        FlashInferMLASparseSM120Backend as Backend,
    )
    from vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm120 import (
        FlashInferMLASparseSM120Impl as Impl,
    )
    from vllm.model_executor.layers.attention.mla_attention import MLAAttention
    from vllm.v1.kv_cache_interface import MLAAttentionSpec
    from vllm.v1.attention.backend import MultipleOf

    if getattr(Impl, "_nq_fp4_kv", False):
        return
    if not getattr(Impl, "_nq_nope_padding", False):
        raise RuntimeError("Install the Flash NoPE adapter before FP4 cache support")
    old_page = MLAAttentionSpec.real_page_size_bytes.fget
    old_spec = MLAAttention.get_kv_cache_spec
    old_init = Impl.__init__

    def shape(num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str="auto"):
        if head_size != LATENT or num_kv_heads != 1 or cache_dtype_str != "fp8_ds_mla":
            raise ValueError(
                "FP4 override requires byte-packed NoPE MLA, head size 512"
            )
        return (num_blocks, block_size, BYTES)

    def page(self):
        # Platform._align_hybrid_block_size probes an *unbound* one-token MLA
        # spec before layers exist. It too must use 320 B/token, otherwise KDA
        # padding can erase the cache savings (or force incompatible pages).
        alignment_probe = (
            self.block_size == 1
            and self.head_size == LATENT
            and self.model_version is None
            and self.compress_ratio == 1
            and self.cache_dtype_str in ("fp8", "fp8_e4m3", "fp8_ds_mla")
        )
        if self.model_version == FORMAT or alignment_probe:
            if (
                self.head_size != LATENT
                or self.compress_ratio != 1
                or self.num_kv_heads != 1
            ):
                raise ValueError("Incompatible FP4 MLA spec")
            return self.storage_block_size * BYTES
        return old_page(self)

    def spec(self, config):
        result = old_spec(self, config)
        if self.attn_backend is not Backend:
            return result
        if getattr(self, "prefill_backend", None) is not None:
            raise ValueError(
                "FP4 MLA requires sparse MQA prefill; dense prefill cache readers are incompatible"
            )
        if type(result) is not MLAAttentionSpec or result.model_version is not None:
            raise ValueError(
                "FP4 MLA does not support sliding/compressed or pre-stamped specs"
            )
        return replace(result, model_version=FORMAT, indexes_kv_by_block_stride=True)

    def checked_init(self, *args, **kwargs):
        old_init(self, *args, **kwargs)
        from vllm.config import get_current_vllm_config

        c = get_current_vllm_config()
        if (
            getattr(c.model_config.hf_text_config, "model_type", None)
            != "glm5_next_text"
            or not self._nq_nope
            or c.parallel_config.tensor_parallel_size != 1
            or c.parallel_config.decode_context_parallel_size != 1
            or not c.model_config.enforce_eager
            or c.scheduler_config.max_num_seqs != 1
            or c.cache_config.enable_prefix_caching
            or c.kv_transfer_config is not None
        ):
            raise ValueError(
                "FP4 MLA requires Flash TP1/DCP1 eager, max-num-seqs 1, no prefix cache or KV connector"
            )
        if (
            c.speculative_config is not None
            and c.speculative_config.num_speculative_tokens not in (1, 2)
        ):
            raise ValueError("FP4 Flash integration only supports MTP1 or MTP2")

    def update(
        self, kv_c_normed, k_pe, kv_cache, slot_mapping, kv_cache_dtype, k_scale
    ):
        if k_pe.shape[-1] != 0:
            raise ValueError("FP4 Flash MLA requires empty rotary keys")
        if kv_cache.numel():
            pack(kv_c_normed, kv_cache, slot_mapping.flatten())

    def forward(self, q, kv_cache, metadata, layer):
        from vllm.v1.attention.backends.mla.sparse_utils import (
            triton_convert_req_index_to_global_index,
        )

        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)
        n = len(q)
        idx = self.topk_indices_buffer[:n]
        if idx.shape[1] not in (2048, 2176):
            raise ValueError("Unexpected Flash sparse window width")
        physical = triton_convert_req_index_to_global_index(
            metadata.req_id_per_token[:n],
            metadata.block_table,
            idx,
            BLOCK_SIZE=metadata.block_size,
            NUM_TOPK_TOKENS=idx.shape[1],
        )
        # The published DSA+SWA buffer already masks causality and removes
        # duplicates between the two sets. One softmax covers their union.
        return attention(q, kv_cache, physical, self.scale), None

    Backend.get_kv_cache_shape = staticmethod(shape)
    # Keep the manager page intact; virtual 64/256-token splitting loses the
    # location of padding between larger hybrid pages. Our kernels accept any
    # multiple of 64 and use the actual physical page stride.
    Backend.get_supported_kernel_block_sizes = staticmethod(lambda: [MultipleOf(64)])
    Backend.indexes_kv_by_block_stride = staticmethod(lambda: True)
    MLAAttentionSpec.real_page_size_bytes = property(page)
    MLAAttention.get_kv_cache_spec = spec
    Impl.__init__ = checked_init
    Impl.do_kv_cache_update = update
    Impl.forward_mqa = forward
    Impl._nq_fp4_kv = True
    print(
        "EXPERIMENTAL Flash FP4 MLA: E2M1/g16 + FP16 scales, 320 B/token; "
        "indexer and KDA unchanged. Quality and GPU performance unvalidated.",
        flush=True,
    )
