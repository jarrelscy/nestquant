"""Batched layer-major prefill: attention in large chunks, then each routed expert once per layer.

layer_major_prefill in native_generate.py runs the prompt through each layer in chunks of at most 16
tokens and computes one banked matvec per (token, expert) pair, so every routed token decodes its
experts' packed weights again. Here each routed layer runs in three phases over the whole prompt:

A. For each attention chunk (thousands of tokens): attention, the FFN hyper-connection collapse and
   norm, and the router. Sparse-attention layers use a head-grouped implementation so the projected
   keys and values of a long context never exist for all heads at once.
B. The router output for all prompt tokens is grouped by expert. Each used expert's dense rotated
   weights (base plus 4-bit refinement) are decoded once and its tokens go through GPU matmuls
   (prefill_kernels). Refinements come from the resident decode bank when present; the rest are
   streamed from the checkpoint ahead of use by prefill_stream, overlapping SSD reads with compute.
C. For each chunk: the shared expert and the hyper-connection expansion.

Every routed expert runs with its 4-bit refinement, as in the existing prefill. Only the predictor's
last `predictor_tail` prompt rows are committed (all of them for prompts up to that length): its
policies keep an EMA with a 64-row half-life and its network attends over 256 rows, and committing
tens of thousands of rows one at a time costs about 3 ms each on the CPU.
"""

from functools import partial
import time

import mlx.core as mx
import numpy as np
from mlx_vlm.models.base import create_ssm_mask
from mlx_vlm.models.deepseek_v4.hyper_connection import hc_expand
from mlx_vlm.models.glm5_next import language
from mlx_vlm.models.linear import DECODE_BLOCK_SIZE, linear
from mlx_vlm.models.sparse_attention import indexed_sparse_attention

import prefill_kernels as pk
from prefill_stream import RefinementStaging, RefinementStream, ShardedRefinementReader

COLD = 0xffffffff


def _multi_linear(module, x, heads, transpose):
    """Apply a (possibly quantized) MultiLinear to a slice of its heads."""
    if hasattr(module, 'scales'):
        biases = module.get('biases')
        return mx.quantized_matmul(
            x, module['weight'][heads], scales=module['scales'][heads],
            biases=None if biases is None else biases[heads], transpose=transpose,
            group_size=module.group_size, bits=module.bits, mode=module.mode)
    weight = module.weight[heads]
    return x @ (weight.swapaxes(-1, -2) if transpose else weight)


def _evaluated_pool_select(original, chunk, q, pool_keys, weights, pool_ends, pool_valid, query_positions,
                           select_k, scale, chunk_size=512):
    """language._exact_pool_select over query slices, evaluating each slice before the next.

    The original loops over query chunks inside one lazy graph, so every chunk's float32 score
    temporaries stay alive until the end (about 1.4 GiB for 2048 queries at 45K keys). Each slice runs
    the original on its own queries, so the selection is unchanged.
    """
    length = query_positions.shape[0]
    selected, valid = [], []
    for start in range(0, length, chunk):
        end = min(start + chunk, length)
        part = original(None if q is None else q[:, start:end], pool_keys,
                        None if weights is None else weights[:, start:end], pool_ends, pool_valid,
                        query_positions[start:end], select_k, scale, chunk_size=end - start)
        mx.eval(part)
        selected.append(part[0])
        valid.append(part[1])
    if len(selected) == 1:
        return selected[0], valid[0]
    return mx.concatenate(selected, axis=1), mx.concatenate(valid, axis=1)


def _routed_layer(layer):
    # native_generate.PagedExperts; checked by attribute because the script may run as __main__.
    return hasattr(layer.mlp, '_bank') and hasattr(layer.mlp, 'gate')


class SparseAttentionPrefill:
    """Glm5NextAttention for long prefill chunks without materializing all heads' keys and values.

    Matches Glm5NextAttention.__call__ (projections, latent cache update and indexer) and replaces its
    prefill attention: keys and values are projected from the latent cache for a group of heads at a
    time. Short contexts use fused SDPA with a mask built from the indexer's selection, which computes
    the same softmax over the same keys as the indexed kernel.

    Long contexts use the absorbed form that decode uses (queries through embed_q, attention over the
    512-wide latents, outputs through unembed_out), where all heads share each key. Consecutive queries
    select overlapping pools, so each block of BLOCK queries gathers the union of its selected latents
    once and runs SDPA over it with a per-query mask. The indexed kernel (one threadgroup per head and
    query, no key reuse) is used instead when union_min_keys is None (the default): in the full model
    the union path measured slower (36.5 s vs 30.1 s of attention at 45K tokens) because bounding its
    unfused score temporary forces small SDPA calls.
    """

    BLOCK = 32

    def __init__(self, attention, dense_max_keys=28672, head_budget_bytes=512 * 1024**2,
                 indexer_budget_bytes=384 * 1024**2, union_min_keys=None, scores_budget_bytes=384 * 1024**2):
        self.attention = attention
        self.union_min_keys = union_min_keys
        self.scores_budget_bytes = scores_budget_bytes
        self.indexer_budget_bytes = indexer_budget_bytes
        self.dense_max_keys = dense_max_keys
        self.head_budget_bytes = head_budget_bytes
        self.stats = dict(dense_calls=0, sparse_calls=0, union_calls=0, indexer_seconds=0.0, attention_seconds=0.0)

    def __call__(self, x, padding_mask=None, cache=None, prev_topk_indices=None):
        attn = self.attention
        batch, length, _ = x.shape
        if length <= DECODE_BLOCK_SIZE or cache is None or batch != 1:
            return attn(x, padding_mask, cache, prev_topk_indices)
        q_a, kv_a = mx.split(linear(attn.qkv_a_proj, x), (attn.q_lora_rank,), axis=-1)
        q_resid = attn.q_a_layernorm(q_a)
        q = linear(attn.q_b_proj, q_resid).reshape(batch, length, attn.num_heads, attn.q_head_dim)
        q = q.transpose(0, 2, 1, 3)
        new_latent = attn.kv_a_layernorm(kv_a)[:, None]
        kv_cache = cache[0]
        offset = kv_cache.offset
        latent, _ = kv_cache.update_and_fetch(new_latent, mx.zeros((batch, 1, length, 0), dtype=new_latent.dtype))
        mx.eval(q, latent)
        started = time.perf_counter()
        if attn.indexer is None:
            if prev_topk_indices is None:
                raise ValueError('Shared indexer layer has no previous top-k indices.')
            topk = prev_topk_indices
        else:
            # The indexer scores every pool for a chunk of queries at once in float32, twice over
            # (scores and weighted scores). Bound that temporary for long contexts.
            pools = max(1, (offset + length) // attn.indexer.index_kpool)
            chunk = int(min(512, max(32, self.indexer_budget_bytes // (2 * 4 * attn.indexer.n_heads * pools))))
            original = language._exact_pool_select
            language._exact_pool_select = partial(_evaluated_pool_select, original, chunk)
            try:
                topk = attn.indexer(x, q_resid, padding_mask, cache[1], cache[2], offset)
                mx.eval(topk)
            finally:
                language._exact_pool_select = original
        indexed = time.perf_counter()
        self.stats['indexer_seconds'] += indexed - started
        keys_total = latent.shape[2]
        if self.union_min_keys is not None and keys_total > max(self.union_min_keys, self.dense_max_keys):
            self.stats['union_calls'] += 1
            out = self._union_attention(q, latent, topk)
            self.stats['attention_seconds'] += time.perf_counter() - indexed
            out = out.transpose(0, 2, 1, 3).reshape(batch, length, -1)
            return linear(attn.o_proj, out), topk
        dense = keys_total <= self.dense_max_keys
        mask = None
        if dense:
            safe = mx.where(topk >= 0, topk, keys_total)
            mask = mx.zeros((batch, length, keys_total + 1), dtype=mx.bool_)
            mask = mx.put_along_axis(mask, safe, mx.array(True), axis=-1)[..., :keys_total][:, None]
            self.stats['dense_calls'] += 1
        else:
            self.stats['sparse_calls'] += 1
        per_head = keys_total * (attn.q_head_dim + attn.v_head_dim) * latent.dtype.size
        group = attn.num_heads
        while group > 1 and group * per_head > self.head_budget_bytes:
            group //= 2
        outputs = []
        for start in range(0, attn.num_heads, group):
            heads = slice(start, start + group)
            keys = _multi_linear(attn.embed_q, latent, heads, transpose=False)
            values = _multi_linear(attn.unembed_out, latent, heads, transpose=True)
            queries = q[:, heads]
            if dense:
                out = mx.fast.scaled_dot_product_attention(queries, keys, values, scale=attn.scale, mask=mask)
            else:
                out = indexed_sparse_attention(queries, keys, values, topk, attn.scale)
                if out is None:
                    raise RuntimeError('Indexed sparse attention is unavailable for this prefill chunk')
            if group < attn.num_heads:
                mx.eval(out)
            outputs.append(out)
        out = outputs[0] if len(outputs) == 1 else mx.concatenate(outputs, axis=1)
        mx.eval(out)
        self.stats['attention_seconds'] += time.perf_counter() - indexed
        out = out.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return linear(attn.o_proj, out), topk


    def _union_attention(self, q, latent, topk):
        attn = self.attention
        _, heads, length, _ = q.shape
        keys_total = latent.shape[2]
        rank = latent.shape[-1]
        block = self.BLOCK
        blocks = (length + block - 1) // block
        padded = blocks * block
        absorbed = _multi_linear(attn.embed_q, q, slice(None), transpose=True)[0]    # (heads, length, rank)
        if padded > length:
            absorbed = mx.concatenate([absorbed, mx.zeros((heads, padded - length, rank), absorbed.dtype)], axis=1)
        indices = topk[0]
        if padded > length:
            indices = mx.concatenate([indices, mx.full((padded - length, indices.shape[1]), -1, indices.dtype)])
        safe = mx.where(indices >= 0, indices, keys_total)
        selected = mx.zeros((padded, keys_total + 1), dtype=mx.bool_)
        selected = mx.put_along_axis(selected, safe, mx.array(True), axis=-1)[:, :keys_total]
        selected = selected.reshape(blocks, block, keys_total)
        used = selected.any(axis=1)                                                  # (blocks, keys)
        counts = used.sum(axis=-1)
        mx.eval(absorbed, selected, used, counts)
        counts_host = np.asarray(counts)
        positions = mx.arange(keys_total, dtype=mx.int32)
        outputs = []
        start = 0
        while start < blocks:
            # Bound the unfused score temporary (sub, heads, block, union) in float32.
            width = int(counts_host[start:start + 8].max())
            sub = max(1, min(blocks - start, self.scores_budget_bytes // (4 * heads * block * max(width, 1))))
            end = start + sub
            union = int(counts_host[start:end].max())
            part = used[start:end]
            # Compact each block's selected key positions in ascending order.
            slot = mx.where(part, mx.cumsum(part.astype(mx.int32), axis=-1) - 1, union)
            order = mx.zeros((sub, union + 1), dtype=mx.int32)
            order = mx.put_along_axis(order, slot, mx.broadcast_to(positions[None], (sub, keys_total)), axis=-1)
            order = order[:, :union]
            valid = mx.arange(union)[None] < counts[start:end, None]
            keys = latent[0, 0][order]                                               # (sub, union, rank)
            mask = mx.take_along_axis(selected[start:end], order[:, None, :], axis=-1) & valid[:, None, :]
            queries = absorbed[:, start * block:end * block].reshape(heads, sub, block, rank).transpose(1, 0, 2, 3)
            out = mx.fast.scaled_dot_product_attention(queries, keys[:, None], keys[:, None], scale=attn.scale,
                                                       mask=mask[:, None])
            out = out.transpose(1, 0, 2, 3).reshape(heads, sub * block, rank)
            mx.eval(out)
            outputs.append(out)
            start = end
        out = mx.concatenate(outputs, axis=1)[:, :length] if len(outputs) > 1 else outputs[0][:, :length]
        out = _multi_linear(attn.unembed_out, out[None], slice(None), transpose=True)  # (1, heads, length, v)
        mx.eval(out)
        return out


class BatchedPrefill:
    """Callable with the layer_major_prefill signature: (model, chunks, cache, guard, hot) -> logits."""

    def __init__(self, root, attention_chunk=4096, staging_rows=16, readers=16, group_experts=4,
                 group_rows=3072, predictor_tail=512, dense_max_keys=28672, record=None,
                 cache_limit_bytes=512 * 1024**2, long_context=16384, long_attention_chunk=2048,
                 linear_chunk=1024, long_group_experts=2, long_group_rows=2048):
        self.root = root
        self.attention_chunk = attention_chunk
        self.staging_rows = staging_rows
        self.readers = readers
        self.group_experts = group_experts
        self.group_rows = group_rows
        self.predictor_tail = predictor_tail
        self.dense_max_keys = dense_max_keys
        self.record = record
        # The runtime caps MLX's buffer cache at 256 MiB. Prefill frees and reallocates hundreds of MiB of
        # scratch per expert group and attention chunk; above the cap every allocation gets fresh,
        # zero-filled pages from the kernel. A larger cache during prefill reuses those buffers instead.
        self.cache_limit_bytes = cache_limit_bytes
        # Beyond `long_context` prompt tokens the residual stream, MoE input and accumulator dominate memory,
        # so attention runs in smaller chunks to shrink its per-chunk temporaries.
        self.long_context = long_context
        self.long_attention_chunk = long_attention_chunk
        self.linear_chunk = linear_chunk
        self.long_group_experts = long_group_experts
        self.long_group_rows = long_group_rows
        self.reader = None
        self.staging = None
        self.report = None
        self.__name__ = 'batched_layer_major_prefill'

    def close(self):
        if self.reader is not None:
            self.reader.close()
            self.reader = None
        if self.staging is not None:
            self.staging.close()
            self.staging = None

    def _ensure_staging(self):
        if self.reader is None:
            self.reader = ShardedRefinementReader(self.root)
            self.reader.manifest(3)
        if self.staging is None:
            self.staging = RefinementStaging(self.reader.geometry, self.staging_rows)

    def __call__(self, model, chunks, cache, guard, hot, capture=None):
        """capture: optional (layer ids, collapse function). The collapsed output of each listed layer over the
        whole prompt is left in self.captured[layer] for block drafters."""
        self.captured = {}
        wanted = set(capture[0]) if capture else set()
        collapse = capture[1] if capture else None

        def keep(number, values):
            if number in wanted:
                self.captured[number] = mx.concatenate([collapse(value) for value in values], axis=1)
                mx.eval(self.captured[number])

        ids = [token for chunk in chunks for token in chunk]
        total = len(ids)
        size = self.attention_chunk if total <= self.long_context else min(self.attention_chunk,
                                                                            self.long_attention_chunk)
        spans = [(start, min(start + size, total)) for start in range(0, total, size)]
        timings = dict(linear_attention=0.0, sparse_attention=0.0, attention=0.0, moe=0.0, combine=0.0, predictor=0.0, dense_layers=0.0,
                       moe_wait=0.0, moe_host=0.0)
        moe_stats = dict(groups=0, decode_bank_experts=0, streamed_experts=0, rows=0, unused_experts=0)
        routed = [layer.mlp for layer in model.layers if _routed_layer(layer)]
        if any(not mlp._bank.fully_resident for mlp in routed):
            raise ValueError('Batched prefill requires the fully resident base bank')
        stream = None
        decode_banks = hot.banks if hot is not None else {}
        if hot is not None:
            self._ensure_staging()
            keys = [(mlp._bank.layer, expert) for mlp in routed for expert in range(288)
                    if expert not in decode_banks[mlp._bank.layer].slots]
            stream = RefinementStream(self.reader, self.staging, keys, self.readers)
        tail_start = max(0, total - self.predictor_tail) if hot is not None else total
        trace = {}
        per_layer = []
        attention_wrappers = {}
        started = time.perf_counter()
        phase_peaks = dict(attention=0, moe=0, combine=0)
        baseline = mx.get_active_memory()
        from resources import process_usage
        footprint_at_start = process_usage().phys_footprint
        mx.clear_cache()
        previous_cache_limit = mx.set_cache_limit(self.cache_limit_bytes)
        try:
            hidden = [mx.repeat(model.model.embed_tokens(mx.array([ids[a:b]]))[:, :, None],
                                model.config.hc_mult, axis=2) for a, b in spans]
            indices = [None] * len(spans)
            for number, (layer, state) in enumerate(zip(model.layers, cache, strict=True)):
                layer_started = time.perf_counter()
                linear_layer = layer.block_type == 'linear_attention'
                if not _routed_layer(layer):
                    for i, value in enumerate(hidden):
                        mask = create_ssm_mask(value[:, :, 0], state) if linear_layer else None
                        hidden[i], indices[i] = layer(value, mask, state, indices[i])
                        mx.eval(hidden[i], indices[i], state.state)
                        guard.check()
                    timings['dense_layers'] += time.perf_counter() - layer_started
                    keep(number, hidden)
                    continue
                mlp = layer.mlp
                if not linear_layer and number not in attention_wrappers:
                    attention_wrappers[number] = SparseAttentionPrefill(
                        layer.self_attn, self.dense_max_keys, head_budget_bytes=384 * 1024**2,
                        indexer_budget_bytes=256 * 1024**2)
                # Phase A: attention, FFN collapse and norm, and routing for every chunk.
                mx.clear_cache()
                mx.reset_peak_memory()
                collapsed, posts, combs, routes, weights = [], [], [], [], []
                for i, value in enumerate(hidden):
                    if linear_layer:
                        # The linear-attention branch holds several projection-sized temporaries per token;
                        # run it in sub-chunks (the recurrent state carries across them).
                        parts = []
                        for start in range(0, value.shape[1], self.linear_chunk):
                            part = value[:, start:start + self.linear_chunk]
                            mask = create_ssm_mask(part[:, :, 0], state)
                            part = layer.attn_hc.apply_branch(part, layer.input_layernorm, layer.self_attn, mask,
                                                              state)
                            mx.eval(part, state.state)
                            parts.append(part)
                        value = parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=1)
                        del parts
                    else:
                        value, indices[i] = layer.attn_hc.apply_branch(
                            value, layer.input_layernorm, attention_wrappers[number], None, state, indices[i])
                    x, post, comb = layer.ffn_hc(value)
                    x = layer.post_attention_layernorm(x)
                    route, weight = mlp.gate(x)
                    hidden[i] = value
                    mx.eval(value, x, post, comb, route, weight, state.state,
                            indices[i] if indices[i] is not None else [])
                    collapsed.append(x)
                    posts.append(post)
                    combs.append(comb)
                    routes.append(route)
                    weights.append(weight)
                    guard.check()
                phase_b = time.perf_counter()
                phase_peaks['attention'] = max(phase_peaks['attention'], mx.get_peak_memory() - baseline)
                mx.clear_cache()
                mx.reset_peak_memory()
                timings['attention'] += phase_b - layer_started
                timings['linear_attention' if linear_layer else 'sparse_attention'] += phase_b - layer_started
                # Phase B: every routed expert once over all prompt tokens.
                X = mx.concatenate(collapsed, axis=1)[0] if len(collapsed) > 1 else collapsed[0][0]
                collapsed = None
                route_ids = np.concatenate([np.asarray(r).reshape(-1, r.shape[-1]) for r in routes])
                route_weights = mx.concatenate([w.reshape(-1) for w in weights])
                if tail_start < total:
                    tail_x = X[tail_start:]
                    norms = mx.square(tail_x.astype(mx.float32)).sum(axis=-1)
                    tail_w = route_weights.reshape(total, -1)[tail_start:].astype(mx.float32)
                    trace[mlp._bank.layer] = (route_ids[tail_start:].reshape(-1), np.asarray(tail_w),
                                              np.asarray(norms))
                Y = self._routed(mlp, X, route_ids, route_weights, decode_banks.get(mlp._bank.layer), stream,
                                 guard, timings, moe_stats)
                phase_c = time.perf_counter()
                phase_peaks['moe'] = max(phase_peaks['moe'], mx.get_peak_memory() - baseline)
                mx.clear_cache()
                mx.reset_peak_memory()
                timings['moe'] += phase_c - phase_b
                per_layer.append((number, round(phase_b - layer_started, 3), round(phase_c - phase_b, 3)))
                # Phase C: shared expert and hyper-connection expansion.
                for i, (a, b) in enumerate(spans):
                    x = X[a:b][None]
                    output = Y[a:b].astype(x.dtype)[None] + mlp.shared_experts(x)
                    hidden[i] = hc_expand(output, hidden[i], posts[i], combs[i])
                    mx.eval(hidden[i])
                    posts[i] = combs[i] = None
                    guard.check()
                del X, Y
                keep(number, hidden)
                phase_peaks['combine'] = max(phase_peaks['combine'], mx.get_peak_memory() - baseline)
                timings['combine'] += time.perf_counter() - phase_c
                if self.record is not None:
                    self.record('prefill_layer_complete', layer=number, seconds=time.perf_counter() - layer_started)
            value = model.model.norm(hidden[-1][:, -1:].mean(axis=2))
            logits = (model.model.embed_tokens.as_linear(value) if model.config.tie_word_embeddings
                      else model.lm_head(value))
            mx.eval(logits)
            hidden = None
            if hot is not None:
                predictor_started = time.perf_counter()
                self._commit(hot, ids, tail_start, trace)
                timings['predictor'] = time.perf_counter() - predictor_started
        finally:
            if stream is not None:
                stream.close()
            mx.set_cache_limit(previous_cache_limit)
            mx.clear_cache()
        attention_stats = dict(dense_calls=0, sparse_calls=0, union_calls=0, indexer_seconds=0.0,
                               attention_seconds=0.0)
        for wrapper in attention_wrappers.values():
            for name in attention_stats:
                attention_stats[name] += wrapper.stats[name]
        self.report = dict(footprint_at_start_bytes=footprint_at_start, prompt_tokens=total, attention_chunk=size, seconds=time.perf_counter() - started,
                           timings=timings, moe=moe_stats, attention=attention_stats,
                           predictor_rows=total - tail_start, phase_peak_bytes_over_start=phase_peaks, per_layer_attention_moe_seconds=per_layer,
                           stream=dict(stream.stats) if stream is not None else None,
                           staging_bytes=self.staging.bytes if self.staging is not None else 0)
        return logits

    def _routed(self, mlp, X, route_ids, route_weights, decode_bank, stream, guard, timings, moe_stats):
        host_started = time.perf_counter()
        bank = mlp._bank
        layer = bank.layer
        total, top_k = route_ids.shape
        flat = route_ids.reshape(-1)
        counts = np.bincount(flat, minlength=288)
        order = np.argsort(flat, kind='stable').astype(np.uint32)
        starts = np.concatenate([[0], np.cumsum(counts)])
        used = counts > 0
        moe_stats['rows'] += int(flat.size)
        moe_stats['unused_experts'] += int((~used).sum())
        if decode_bank is not None:
            resident = [e for e in range(288) if used[e] and e in decode_bank.slots]
            streamed = [e for e in range(288) if e not in decode_bank.slots]
            stream.skip([(layer, e) for e in streamed if not used[e]])
            streamed = [e for e in streamed if used[e]]
        else:
            resident = [e for e in range(288) if used[e]]
            streamed = []
        moe_stats['decode_bank_experts'] += len(resident)
        moe_stats['streamed_experts'] += len(streamed)
        # Each group's down-projection kernel adds its rows into Y in place (see prefill_kernels).
        Y = mx.zeros((total, X.shape[1]), dtype=mx.float32)
        mx.eval(Y)
        markers = []
        pending = None
        base = bank.planes

        long_prompt = total > self.long_context
        group_experts = self.long_group_experts if long_prompt else self.group_experts
        group_rows = self.long_group_rows if long_prompt else self.group_rows

        def groups(experts):
            group, rows = [], 0
            for expert in experts:
                if group and (len(group) >= group_experts or rows + counts[expert] > group_rows):
                    yield group
                    group, rows = [], 0
                group.append(expert)
                rows += counts[expert]
            if group:
                yield group

        sources = [('decode', resident)] + ([('stream', streamed)] if streamed else [])
        timings['moe_host'] += time.perf_counter() - host_started
        for source, experts in sources:
            for group in groups(experts):
                host_started = time.perf_counter()
                if source == 'stream':
                    wait_started = time.perf_counter()
                    slots = [stream.take((layer, e)) for e in group]
                    timings['moe_wait'] += time.perf_counter() - wait_started
                    refinements = self.staging.planes
                elif decode_bank is not None:
                    slots = [decode_bank.slots[e] for e in group]
                    refinements = decode_bank.planes
                else:
                    slots, refinements = None, None
                rows = np.concatenate([order[starts[e]:starts[e + 1]] for e in group])
                locals_ = np.repeat(np.arange(len(group), dtype=np.uint32), counts[group])
                experts_mx = mx.array(np.asarray(group, dtype=np.uint32))
                slots_mx = None if slots is None else mx.array(np.asarray(slots, dtype=np.uint32))
                dense = {name: pk.dense_rotated(planes, meta, experts_mx, slots_mx,
                                                None if refinements is None else refinements[name])
                         for name, (planes, meta) in base.items()}
                params = pk.group_parameters(base, experts_mx, slots_mx, refinements)
                rows_mx = mx.array(rows)
                tokens = rows_mx // top_k
                done = pk.expert_group_forward(X, tokens, mx.array(locals_), route_weights[rows_mx], dense, params,
                                               accumulator=Y)
                markers.append(done)
                consumed = list(dense.values()) + [value for entry in params.values() for value in entry.values()
                                                   if isinstance(value, mx.array)]
                built = time.perf_counter()
                timings['moe_build'] = timings.get('moe_build', 0.0) + built - host_started
                mx.async_eval(done)
                timings['moe_submit'] = timings.get('moe_submit', 0.0) + time.perf_counter() - built
                timings['moe_host'] += time.perf_counter() - host_started
                if pending is not None:
                    self._retire(pending, stream, timings)
                pending = (consumed, slots if source == 'stream' else None)
                moe_stats['groups'] += 1
                guard.check()
        if pending is not None:
            self._retire(pending, stream, timings)
        wait_started = time.perf_counter()
        mx.eval(markers)
        mx.synchronize()
        timings['moe_wait'] += time.perf_counter() - wait_started
        return Y

    @staticmethod
    def _retire(pending, stream, timings):
        consumed, slots = pending
        started = time.perf_counter()
        # The dense weights and gathered parameters are the only readers of the staging rows.
        mx.eval(consumed)
        timings['moe_wait'] += time.perf_counter() - started
        if slots is not None:
            stream.release(slots)

    @staticmethod
    def _commit(hot, ids, tail_start, trace):
        if not trace:
            return
        if tail_start:
            # Start the predictor clocks at the first committed row so positions match a full commit. A
            # suffix prefill on top of earlier committed tokens starts from the current clock.
            base = hot.position
            predictor = hot.predictor
            predictor.t = base + tail_start
            predictor.prepared_t = -1
            predictor.net.position = base + tail_start
            # The policies keep their own row count only for the refresh cadence; leaving it at zero makes
            # the first committed row skip the refresh, which needs the previous row's jT block.
            hot.position = base + tail_start
        hot.commit(ids[tail_start:], trace=trace)
