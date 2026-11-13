"""Faster short-block forward (1-8 tokens with a cache) for the GLM-5.3-Flash backbone.

Used for single-token decode and for speculative verify blocks. Measured on the backbone with routed
experts removed (profile_backbone.py), most of a step is weight streaming; the rest is serial small
kernels and host graph construction during which the GPU idles. Changes, each limited to
floating-point reassociation:

* Hyper-connections: each sublayer's collapse used seven kernels (cast, RMS norm, FP32 mix GEMV,
  Sinkhorn collapse, input RMS norm, expand GEMM, expand elementwise). Here the expand of one sublayer
  and the mix partial sums of the next share one kernel, and one more kernel reduces the partials, runs
  Sinkhorn, collapses and applies the sublayer's RMS norm: two kernels per sublayer instead of seven.
* Early GPU start: every ASYNC_EVERY layers the residual streams are submitted with mx.async_eval, so
  the GPU runs the first layers while Python builds the rest of the graph.
* Sparse attention: the latent is shared by all heads, so all queries of a block attend in one pass
  with heads in the matmul rows, and embed_q/unembed_out run once per block instead of once per token.
* Narrow (128-input) quantized projections of a block run as one matmul (bit-identical rows).

`install(model)` switches the text model and its sparse attention layers to this path. Prefill, batched
inputs and any attention mask keep the original code.
"""

from contextlib import contextmanager
from functools import lru_cache

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models import linear as linear_module
from mlx_vlm.models.glm5_next import language


MAX_BLOCK = 8          # longest token block (decode or speculative verify) that takes the fast path
ASYNC_EVERY = 4        # start the GPU on the first layers while later layers are still being built
_PARTIAL_COLUMNS = 64  # threads (and columns) per partial threadgroup
_PARTIAL_STRIDE = 32   # floats per partial row: 24 mix dot products, the sum of squares, padding


def _hc_partial_source(has_expand):
    expand = """
        out += row * HC * D;
        float od = static_cast<float>(o[row * D + d]);
        float c[HC * HC];
        for (int k = 0; k < HC * HC; ++k) c[k] = comb[row * HC * HC + k];
        float xin[HC];
        for (int j = 0; j < HC; ++j) xin[j] = static_cast<float>(x[j * D + d]);
        for (int i = 0; i < HC; ++i) {
            float m = c[i] * xin[0];
            for (int j = 1; j < HC; ++j) m = fma(c[j * HC + i], xin[j], m);
            T value = static_cast<T>(post[row * HC + i] * od + m);
            out[i * D + d] = value;
            v[i] = static_cast<float>(value);
        }
    """ if has_expand else """
        for (int i = 0; i < HC; ++i) v[i] = static_cast<float>(x[i * D + d]);
    """
    return f"""
        uint tid = thread_position_in_threadgroup.x;
        uint tg = threadgroup_position_in_grid.x;
        uint row = threadgroup_position_in_grid.y;
        uint d = tg * {_PARTIAL_COLUMNS} + tid;
        x += row * HC * D;
        partial += row * NT * {_PARTIAL_STRIDE};
        float v[HC];
        {expand}
        constexpr int MIX = (2 + HC) * HC;
        float acc[MIX + 1];
        float sq = 0.0f;
        for (int i = 0; i < HC; ++i) sq += v[i] * v[i];
        acc[MIX] = sq;
        for (int r = 0; r < MIX; ++r) {{
            const device float* row = fn + r * (HC * D) + d;
            float a = 0.0f;
            for (int i = 0; i < HC; ++i) a += v[i] * row[i * D];
            acc[r] = a;
        }}
        threadgroup float shared[{_PARTIAL_COLUMNS // 32}][MIX + 1];
        uint lane = tid % 32, sg = tid / 32;
        for (int r = 0; r <= MIX; ++r) {{
            float s = simd_sum(acc[r]);
            if (lane == 0) shared[sg][r] = s;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid <= (uint)MIX) {{
            float s = 0.0f;
            for (int g = 0; g < {_PARTIAL_COLUMNS // 32}; ++g) s += shared[g][tid];
            partial[tg * {_PARTIAL_STRIDE} + tid] = s;
        }}
    """


@lru_cache(maxsize=None)
def _partial_kernel(has_expand):
    inputs = ['o', 'x', 'post', 'comb', 'fn'] if has_expand else ['x', 'fn']
    outputs = ['out', 'partial'] if has_expand else ['partial']
    return mx.fast.metal_kernel(
        name='nqbb_hc_expand_partial' if has_expand else 'nqbb_hc_partial',
        input_names=inputs, output_names=outputs, source=_hc_partial_source(has_expand),
        ensure_row_contiguous=True)


_FINISH_SOURCE = """
    uint tid  = thread_position_in_threadgroup.x;
    uint lane = tid % 32;
    uint sg   = tid / 32;
    uint row  = threadgroup_position_in_grid.x;
    partial += row * NT * STRIDE;
    x += row * HC * D;
    normed += row * D;
    post += row * HC;
    comb += row * HC * HC;
    constexpr int MIX = (2 + HC) * HC;
    constexpr int BASE_OFF = 2 * HC;
    constexpr float EPS = EPS_INT * 1e-9;
    constexpr float NORM_EPS = NORM_EPS_INT * 1e-9;
    constexpr int PER = D / 256;

    threadgroup float totals[MIX + 1];
    threadgroup float pre_shared[HC];
    threadgroup float red[8];

    if (tid <= (uint)MIX) {
        float s = 0.0f;
        for (int t = 0; t < NT; ++t) s += partial[t * STRIDE + tid];
        totals[tid] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0) {
        // rms_norm(flatten(x)) @ fn.T == (flatten(x) @ fn.T) * rsqrt(mean(x^2) + eps)
        const float normalizer = metal::precise::rsqrt(totals[MIX] / (HC * D) + NORM_EPS);
        const float pre_scale  = scale[0];
        const float post_scale = scale[1];
        const float comb_scale = scale[2];

        const float active = (lane < (uint)HC) ? 1.0f : 0.0f;
        const uint  llane  = metal::min(lane, (uint)(HC - 1));

        float pre_z  = totals[llane] * normalizer * pre_scale  + base[llane];
        float post_z = totals[HC + llane] * normalizer * post_scale + base[HC + llane];
        float pre_v  = 1.0f / (1.0f + metal::fast::exp(-pre_z)) + EPS;
        float post_v = 2.0f / (1.0f + metal::fast::exp(-post_z));

        if (lane < (uint)HC) {
            pre_shared[lane] = pre_v;
            post[lane] = post_v;
        }

        uint rb = BASE_OFF + llane * HC;
        float4 m4 = float4(totals[rb], totals[rb + 1], totals[rb + 2], totals[rb + 3]) * normalizer;
        float4 b4 = float4(base[rb], base[rb + 1], base[rb + 2], base[rb + 3]);
        float4 v = (m4 * comb_scale + b4) * active;

        float row_max = metal::max(metal::max(v.x, v.y), metal::max(v.z, v.w));
        float4 e = metal::fast::exp(v - row_max) * active;
        float4 r = e * (1.0f / (e.x + e.y + e.z + e.w + EPS)) + EPS * active;

        float4 col_inv = 1.0f / (float4(simd_sum(r.x), simd_sum(r.y), simd_sum(r.z), simd_sum(r.w)) + EPS);
        r *= col_inv;
        for (int iter = 1; iter < ITERS; ++iter) {
            r *= (1.0f / (r.x + r.y + r.z + r.w + EPS)) * active;
            col_inv = 1.0f / (float4(simd_sum(r.x), simd_sum(r.y), simd_sum(r.z), simd_sum(r.w)) + EPS);
            r *= col_inv;
        }
        if (lane < (uint)HC) {
            comb[lane * HC + 0] = r.x;
            comb[lane * HC + 1] = r.y;
            comb[lane * HC + 2] = r.z;
            comb[lane * HC + 3] = r.w;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float p0 = pre_shared[0], p1 = pre_shared[1], p2 = pre_shared[2], p3 = pre_shared[3];
    float col[PER];
    float sq = 0.0f;
    for (int k = 0; k < PER; ++k) {
        uint d = tid + k * 256;
        float x0 = static_cast<float>(x[d]);
        float x1 = static_cast<float>(x[D + d]);
        float x2 = static_cast<float>(x[2 * D + d]);
        float x3 = static_cast<float>(x[3 * D + d]);
        float value = static_cast<float>(static_cast<T>(fma(p0, x0, fma(p1, x1, fma(p2, x2, p3 * x3)))));
        col[k] = value;
        sq += value * value;
    }
    sq = simd_sum(sq);
    if (lane == 0) red[sg] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (int g = 0; g < 8; ++g) total += red[g];
    const float normalizer2 = metal::precise::rsqrt(total / D + NORM_EPS);
    for (int k = 0; k < PER; ++k) {
        uint d = tid + k * 256;
        normed[d] = w[d] * static_cast<T>(col[k] * normalizer2);
    }
"""


@lru_cache(maxsize=None)
def _finish_kernel():
    return mx.fast.metal_kernel(
        name='nqbb_hc_finish', input_names=['partial', 'scale', 'base', 'x', 'w'],
        output_names=['normed', 'post', 'comb'], source=_FINISH_SOURCE, ensure_row_contiguous=True)


def _rows(x):
    return x.size // (x.shape[-1] * x.shape[-2])


def hc_partial(x, fn):
    *_, hc, dim = x.shape
    nt, rows = dim // _PARTIAL_COLUMNS, _rows(x)
    return _partial_kernel(False)(
        inputs=[x, fn], template=[('T', x.dtype), ('HC', hc), ('D', dim), ('NT', nt)],
        grid=(dim, rows, 1), threadgroup=(_PARTIAL_COLUMNS, 1, 1),
        output_shapes=[(rows, nt, _PARTIAL_STRIDE)], output_dtypes=[mx.float32])[0]


def hc_expand_partial(o, x, post, comb, fn):
    *lead, hc, dim = x.shape
    nt, rows = dim // _PARTIAL_COLUMNS, _rows(x)
    out, partial = _partial_kernel(True)(
        inputs=[o, x, post, comb, fn], template=[('T', x.dtype), ('HC', hc), ('D', dim), ('NT', nt)],
        grid=(dim, rows, 1), threadgroup=(_PARTIAL_COLUMNS, 1, 1),
        output_shapes=[x.shape, (rows, nt, _PARTIAL_STRIDE)], output_dtypes=[x.dtype, mx.float32])
    return out, partial


def hc_finish(partial, hc_module, x, norm):
    *lead, hc, dim = x.shape
    return _finish_kernel()(
        inputs=[partial, hc_module.scale, hc_module.base, x, norm.weight],
        template=[('T', x.dtype), ('HC', hc), ('D', dim), ('NT', partial.shape[1]), ('STRIDE', _PARTIAL_STRIDE),
                  ('ITERS', hc_module.sinkhorn_iters), ('EPS_INT', round(hc_module.hc_eps / 1e-9)),
                  ('NORM_EPS_INT', round(norm.eps / 1e-9))],
        grid=(256 * _rows(x), 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(*lead, dim), (*lead, hc), (*lead, hc, hc)],
        output_dtypes=[x.dtype, mx.float32, mx.float32])


def _hc_supported(layer, x):
    hc = layer.attn_hc
    return (x.ndim == 4 and x.shape[0] == 1 and 1 <= x.shape[1] <= MAX_BLOCK and x.shape[2] == 4
            and x.shape[3] % 256 == 0 and x.dtype in (mx.bfloat16, mx.float16)
            and hc.fn.dtype == mx.float32 and not layer.training)


def fast_layer(layer, x, partial, mask, cache, topk, next_fn):
    """One decoder layer for a short block. Returns the residual streams, the next sublayer's mix
    partials (or None) and the top-k indices."""
    if partial is None:
        partial = hc_partial(x, layer.attn_hc.fn)
    collapsed, post, comb = hc_finish(partial, layer.attn_hc, x, layer.input_layernorm)
    if layer.is_linear:
        out = layer.self_attn(collapsed, mask, cache)
    else:
        out, topk = layer.self_attn(collapsed, mask, cache, topk)
    x, partial = hc_expand_partial(out, x, post, comb, layer.ffn_hc.fn)
    collapsed, post, comb = hc_finish(partial, layer.ffn_hc, x, layer.post_attention_layernorm)
    out = layer.mlp(collapsed)
    if next_fn is None:
        # Last layer: no following sublayer; the partial sums are computed and discarded.
        x, _ = hc_expand_partial(out, x, post, comb, layer.ffn_hc.fn)
        return x, None, topk
    x, partial = hc_expand_partial(out, x, post, comb, next_fn)
    return x, partial, topk


_ORIGINAL_LINEAR = linear_module.linear


def _block_linear(module, x):
    """Short-block projection. Quantized projections narrower than 512 inputs (the 128-wide gate
    projections of linear attention) otherwise run once per token; one matmul over the block gives
    bit-identical rows for them (checked in test_backbone_fast) with one dispatch instead of 4-8 plus a
    concatenation. (Splitting wide projections of 5-8 row blocks into 4-row groups, which is faster in
    isolation, did not help inside the model.)"""
    if isinstance(module, nn.QuantizedLinear) and x.ndim == 3 and x.shape[-1] % 512:
        return module(x)
    return _ORIGINAL_LINEAR(module, x)


@contextmanager
def _block_linears(enabled):
    if not enabled:
        yield
        return
    saved = language.linear, linear_module.linear
    language.linear = linear_module.linear = _block_linear
    try:
        yield
    finally:
        language.linear, linear_module.linear = saved


class FastTextModel(language.Glm5NextTextModel):
    def __call__(self, inputs, inputs_embeds=None, cache=None, attention_mask=None, hidden_sink=None):
        h = self.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds
        if (cache is None or attention_mask is not None or h.shape[0] != 1 or not 1 <= h.shape[1] <= MAX_BLOCK
                or self.training):
            return super().__call__(inputs, inputs_embeds, cache, attention_mask, hidden_sink)
        h = mx.repeat(h[:, :, None], self.config.hc_mult, axis=2)
        if not _hc_supported(self.layers[0], h):
            return super().__call__(inputs, h[:, :, 0], cache, attention_mask, hidden_sink)
        global _ACTIVE
        _ACTIVE = True
        try:
            return self._fast(h, cache, hidden_sink)
        finally:
            _ACTIVE = False

    def _fast(self, h, cache, hidden_sink):
        # Optional capture of layer outputs for block drafters: (layer ids, sink list, collapse function).
        capture = getattr(self, '_nq_capture_request', None)
        captured = {}
        with _block_linears(h.shape[1] > 1):
            topk = None
            partial = None
            count = len(self.layers)
            for index, (layer, layer_cache) in enumerate(zip(self.layers, cache)):
                mask = None
                if layer.block_type == 'linear_attention':
                    mask = language.create_ssm_mask(h[:, :, 0], layer_cache)
                next_fn = self.layers[index + 1].attn_hc.fn if index + 1 < count else None
                h, partial, topk = fast_layer(layer, h, partial, mask, layer_cache, topk, next_fn)
                if capture is not None and index in capture[0]:
                    captured[index] = capture[2](h)
                if ASYNC_EVERY and (index + 1) % ASYNC_EVERY == 0 and index + 1 < count:
                    mx.async_eval(h, partial)
        if capture is not None:
            capture[1].extend(captured[index] for index in capture[0])
        h = self.norm(h.mean(axis=2))
        if hidden_sink is not None:
            hidden_sink.append(h)
        return h


def _sparse_block_attend(attn, q, latent, topk, kv_cache):
    """Attend a block of queries against their selected latent rows in one pass.

    The latent is shared by all heads (MQA with 512-wide keys), so heads go in the matmul row dimension:
    scores are one [H, 512] x [512, K] product per token. Embedding and unembedding run once for the
    whole block, so each 16 MiB per-layer weight is read once instead of once per token. This is the
    same arithmetic as MLX's composed attention (scaled query, masked precise softmax, value matmul)."""
    batch, heads, length, _ = q.shape
    kv_length = latent.shape[2]
    valid = (topk >= 0) & (topk < kv_length)
    safe = mx.clip(topk, 0, max(kv_length - 1, 0))
    width = safe.shape[-1]
    rank = latent.shape[-1]
    keys = mx.take_along_axis(latent, safe.reshape(batch, 1, length * width, 1), axis=2)
    keys = keys.reshape(batch * length, width, rank)
    query = attn.embed_q(q)  # [B, H, L, R]
    query = query.transpose(0, 2, 1, 3).reshape(batch * length, heads, rank)
    scores = (query * mx.array(attn.scale, dtype=query.dtype)) @ keys.swapaxes(-1, -2)
    scores = mx.where(valid.reshape(batch * length, 1, width), scores, mx.finfo(scores.dtype).min)
    probs = mx.softmax(scores, axis=-1, precise=True)
    output = probs @ keys  # [B*L, H, R]
    output = output.reshape(batch, length, heads, rank).transpose(0, 2, 1, 3)
    return attn.unembed_out(output)


class FastAttention(language.Glm5NextAttention):
    def _attend(self, q, latent, new_latent, topk, kv_cache, projected_cache, last_only=False):
        length = q.shape[2]
        if (_ACTIVE and not last_only and not self.training and q.shape[0] == 1 and 1 <= length <= MAX_BLOCK
                and topk.shape[1] == length):
            out = _sparse_block_attend(self, q, latent, topk, kv_cache)
            return out.transpose(0, 2, 1, 3).reshape(q.shape[0], length, -1), topk
        return super()._attend(q, latent, new_latent, topk, kv_cache, projected_cache, last_only)


_ACTIVE = False


def install(model):
    text = model.model
    if type(text) is language.Glm5NextTextModel:
        text.__class__ = FastTextModel
    for layer in text.layers:
        if type(layer.self_attn) is language.Glm5NextAttention:
            layer.self_attn.__class__ = FastAttention
    return model


def uninstall(model):
    if type(model.model) is FastTextModel:
        model.model.__class__ = language.Glm5NextTextModel
    for layer in model.model.layers:
        if type(layer.self_attn) is FastAttention:
            layer.self_attn.__class__ = language.Glm5NextAttention
    return model


@contextmanager
def patched(model):
    install(model)
    try:
        yield model
    finally:
        uninstall(model)
