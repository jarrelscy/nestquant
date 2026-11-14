"""Sparse-attention microbenchmark and exactness check on the backbone alone (routed experts removed).

Loads the prepared backbone, prefills repository source to each context length with the original code, then for
1/4/8-token blocks captures the inputs of every sparse attention layer from a fast-path forward and times the 11
layers' attention (indexer, attend, whole layer) under each variant, interleaved in one process. Outputs of each
variant are compared with the reference variant layer by layer.

Run inside the GPU lock: /tmp/nq_model_lock.sh kernels python bench_sparse_attention.py --output out.json
"""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import statistics
import time
from unittest.mock import patch

import mlx.core as mx
from mlx_vlm.models.glm5_next import language
from mlx_vlm.speculative.cache_state import start_speculative_cache

import backbone_fast
from native_backbone import load_prepared
from profile_backbone import SharedOnly, install as install_shared

HOME = Path.home()


def variant(name):
    """Patches selecting the sparse attention implementation."""
    stack = ExitStack()
    for part in name.split('+'):
        if part in ('base', 'ref'):
            stack.enter_context(patch.object(backbone_fast, 'FAST_INDEXER', False))

        elif part == 'fastidx':
            stack.enter_context(patch.object(backbone_fast, 'FAST_INDEXER', True))
        elif part == 'noidx':
            stack.enter_context(patch.object(backbone_fast, 'FAST_INDEXER', False))
        elif part == 'new':
            pass
        else:
            raise ValueError(part)
    return stack


def prefill(model, cache, ids, chunk=2048):
    logits = None
    for i in range(0, len(ids), chunk):
        logits = model(mx.array([ids[i:i + chunk]]), cache=cache).logits
        mx.eval(logits, [c.state for c in cache])
    return logits


def capture_inputs(model, cache, tokens):
    """Inputs (x, previous top-k) of every sparse attention layer in one fast forward; cache restored."""
    seen = []
    attends = []
    real = backbone_fast.FastAttention.__call__
    real_attend = backbone_fast.FastAttention._attend

    def hook(self, x, padding_mask=None, cache=None, prev_topk_indices=None, last_only=False):
        out = real(self, x, padding_mask, cache, prev_topk_indices, last_only)
        seen.append((self, x, prev_topk_indices, out[1], attends[-1]))
        return out

    def attend_hook(self, q, latent, new_latent, topk, kv_cache, projected_cache, last_only=False):
        attends.append((q, mx.array(latent), topk))
        return real_attend(self, q, latent, new_latent, topk, kv_cache, projected_cache, last_only)

    transaction = start_speculative_cache(cache, len(tokens))
    with patch.object(backbone_fast.FastAttention, '__call__', hook), \
            patch.object(backbone_fast.FastAttention, '_attend', attend_hook):
        out = model(mx.array([tokens]), cache=cache)
    mx.eval(out.logits, [s[1] for s in seen], [s[3] for s in seen], attends)
    transaction.abort()
    return seen


def attention_layers(model):
    return [(i, layer) for i, layer in enumerate(model.layers) if not layer.is_linear]


def run_stack(model, cache, seen, part):
    """Run all sparse attention layers on captured inputs; return outputs. part: layer|indexer|attend."""
    outs = []
    layers = attention_layers(model)
    for (index, layer), (attn, x, prev, topk_ref, (q, latent, topk_in)) in zip(layers, seen):
        layer_cache = cache[index]
        if part == 'attend':
            outs.append(attn._attend(q, latent, None, topk_in, layer_cache[0], None)[0])
            continue
        transaction = start_speculative_cache(layer_cache, x.shape[1])
        if part == 'layer':
            out, topk = attn(x, None, layer_cache, prev)
            outs.append(out)
        elif part == 'indexer':
            if attn.indexer is None:
                transaction.abort()
                continue
            q_a, kv_a = mx.split(language.linear(attn.qkv_a_proj, x), (attn.q_lora_rank,), axis=-1)
            q_resid = attn.q_a_layernorm(q_a)
            kv_cache = layer_cache[0]
            topk = backbone_fast.run_indexer(attn, x, q_resid, layer_cache[1], layer_cache[2], kv_cache.offset)
            outs.append(topk)
        transaction.abort()
    return outs


_BIG = None


def big_job():
    """A long GPU job queued ahead of a measured graph so that its host encoding overlaps."""
    global _BIG
    if _BIG is None:
        _BIG = mx.random.normal((2048, 2048))
        mx.eval(_BIG)
    y = _BIG
    for _ in range(24):
        y = (y @ _BIG) * 1e-3
    return y


def big_time():
    mx.eval(big_job())
    values = []
    for _ in range(7):
        b = big_job()
        mx.synchronize()
        t = time.perf_counter()
        mx.eval(b)
        values.append(time.perf_counter() - t)
    return statistics.median(values)


def time_parts(model, cache, seen, names, part, repeat):
    """Wall time (host build + GPU, synchronized) and GPU time (host encoding hidden behind a queued job)."""
    times = {name: [] for name in names}
    gpu = {name: [] for name in names}
    outputs = {}
    base = big_time()
    for r in range(repeat):
        for name in (names if r % 2 == 0 else names[::-1]):
            with variant(name):
                backbone_fast._ACTIVE = True
                try:
                    mx.synchronize()
                    t = time.perf_counter()
                    outs = run_stack(model, cache, seen, part)
                    mx.eval(outs)
                    times[name].append(time.perf_counter() - t)
                    outs = run_stack(model, cache, seen, part)
                    b = big_job()
                    mx.synchronize()
                    t = time.perf_counter()
                    mx.async_eval(b)
                    mx.eval(outs, b)
                    gpu[name].append(time.perf_counter() - t - base)
                finally:
                    backbone_fast._ACTIVE = False
                outputs[name] = outs
    med = lambda v: round(1000 * statistics.median(v[2:]), 3)
    return {name: med(v) for name, v in times.items()}, {name: med(v) for name, v in gpu.items()}, outputs


def compare(ref, got):
    diffs = []
    for a, b in zip(ref, got):
        if a.dtype == mx.int32:
            if a.shape != b.shape:  # trimmed padding columns (sparse_kernels.index_select)
                a = mx.concatenate([a[..., :b.shape[-1] - 3], a[..., -3:]], axis=-1)
            diffs.append(float(mx.sum(a != b).item()))
        else:
            diffs.append(float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item()))
    return max(diffs) if diffs else 0.0


def block_logits(model, cache, tokens, name):
    with variant(name):
        transaction = start_speculative_cache(cache, len(tokens))
        out = model(mx.array([tokens]), cache=cache)
        logits = out.logits.astype(mx.float32)
        mx.eval(logits)
        transaction.abort()
    return logits


def time_forward(model, cache, tokens, names, repeat):
    times = {name: [] for name in names}
    for r in range(repeat):
        for name in (names if r % 2 == 0 else names[::-1]):
            with variant(name):
                transaction = start_speculative_cache(cache, len(tokens))
                mx.synchronize()
                t = time.perf_counter()
                out = model(mx.array([tokens]), cache=cache)
                mx.eval(out.logits, [c.state for c in cache])
                times[name].append(time.perf_counter() - t)
                transaction.abort()
    return {name: round(1000 * statistics.median(v[2:]), 3) for name, v in times.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, default=HOME / 'Models/GLM/GLM-5.3-Flash-NestQuant-1.5-4bit')
    parser.add_argument('--backbone', type=Path, default=HOME / 'Models/GLM/backbone-affine4-e8-v1')
    parser.add_argument('--contexts', default='1000,8000,16000,30000')
    parser.add_argument('--blocks', default='1,4,8')
    parser.add_argument('--variants', default='base,new')
    parser.add_argument('--parts', default='layer,indexer')
    parser.add_argument('--repeat', type=int, default=12)
    parser.add_argument('--forward', action='store_true', help='also time whole backbone forwards and compare logits')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    mx.set_cache_limit(512 * 1024**2)
    mx.set_memory_limit(40 * 1024**3)
    model = load_prepared(args.model, args.backbone, lambda *a: None)
    install_shared(model, SharedOnly)
    backbone_fast.install(model)
    from transformers import AutoTokenizer
    import profile_dflash as pdf
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    template = (Path(__file__).resolve().parents[1] / 'spark/flash/thinking_off.jinja').read_text()
    files = sorted(Path(pdf.__file__).resolve().parent.glob('*.py'))
    names = args.variants.split(',')
    results = []
    for context in map(int, args.contexts.split(',')):
        ids = pdf.context_ids(tokenizer, template, files, context)
        cache = model.make_cache()
        with variant('base'):
            logits = prefill(model, cache, ids)
        tokens = [int(logits[0, -1].argmax().item())]
        with variant('base'):
            for _ in range(7):  # a short greedy continuation to verify
                transaction = start_speculative_cache(cache, 1)
                out = model(mx.array([[tokens[-1]]]), cache=cache)
                tokens.append(int(out.logits[0, -1].argmax().item()))
                transaction.abort()
        for block in map(int, args.blocks.split(',')):
            row = dict(context=len(ids), block=block)
            with variant('base'):
                seen = capture_inputs(model, cache, tokens[:block])
            for part in args.parts.split(','):
                times, gpu, outputs = time_parts(model, cache, seen, names, part, args.repeat)
                row[part] = times
                row[part + '_gpu'] = gpu
                row[part + '_maxdiff'] = {name: compare(outputs[names[0]], outputs[name]) for name in names[1:]}
            if args.forward:
                row['forward'] = time_forward(model, cache, tokens[:block], names, args.repeat)
                ref = block_logits(model, cache, tokens[:block], names[0])
                row['logits'] = {}
                for name in names[1:]:
                    got = block_logits(model, cache, tokens[:block], name)
                    row['logits'][name] = dict(
                        max_abs=float(mx.max(mx.abs(ref - got)).item()),
                        mean_abs=float(mx.mean(mx.abs(ref - got)).item()),
                        top1=float(mx.mean((ref.argmax(-1) == got.argmax(-1)).astype(mx.float32)).item()))
            print(json.dumps(row), flush=True)
            results.append(row)
        del cache
        mx.clear_cache()
    args.output.write_text(json.dumps(results, indent=1))


if __name__ == '__main__':
    main()
