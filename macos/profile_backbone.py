"""Profile the decode backbone alone: routed experts removed, gate and shared expert kept.

Loads only the prepared backbone (~5.5 GiB), prefills a prompt, then times forced-token decode steps.
Each variant removes or replaces one component; the difference to the baseline estimates its cost.
Also counts graph primitives per step by op type.
"""

import argparse
from collections import Counter
from contextlib import ExitStack
import io
import json
from pathlib import Path
import re
import statistics
import time
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models.glm5_next import language
from mlx_vlm.models.deepseek_v4 import hyper_connection as hcmod

from native_backbone import RoutedPlaceholder, load_prepared


class SharedOnly(nn.Module):
    """PagedExperts without the routed kernels: router gate, norms and shared expert."""

    def __init__(self, layer):
        super().__init__()
        self.gate = layer.gate
        self.shared_experts = layer.shared_experts
        self.trace = []

    def __call__(self, x):
        ids, weights = self.gate(x)
        tokens = x.reshape(-1, x.shape[-1])
        norms = mx.square(tokens.astype(mx.float32)).sum(axis=-1)
        self.trace.append((ids, weights, norms))
        return self.shared_experts(x)


def count_ops(outputs):
    from mlx.utils import tree_flatten
    arrays = [v for _, v in tree_flatten(outputs) if isinstance(v, mx.array)]
    f = io.StringIO()
    mx.export_to_dot(f, *arrays)
    labels = re.findall(r'label ="([^"]+)", shape=rectangle', f.getvalue())
    return Counter(labels)


def install(model, shared_only):
    for layer in model.layers:
        if isinstance(layer.mlp, (RoutedPlaceholder, SharedOnly)):
            layer.mlp = shared_only(layer.mlp) if shared_only else layer.mlp


def variant_patches(name, model):
    """Context managers that remove one component for timing. Join several with '+'."""
    if '+' in name:
        stack = ExitStack()
        for part in name.split('+'):
            stack.enter_context(variant_patches(part, model))
        return stack
    stack = ExitStack()
    L = language
    if name == 'base':
        pass
    elif name == 'no_shared':
        stack.enter_context(patch.object(SharedOnly, '__call__', lambda self, x: x))
    elif name == 'no_gate':
        stack.enter_context(patch.object(SharedOnly, '__call__', lambda self, x: self.shared_experts(x)))
    elif name == 'no_dense_mlp':
        stack.enter_context(patch.object(L.Glm5NextMLP, '__call__', lambda self, x: x))
        # keep shared experts (also Glm5NextMLP): restore via SharedOnly calling the real implementation
        real = L.Glm5NextMLP.__dict__['__call__']
        stack.enter_context(patch.object(SharedOnly, '__call__',
                                         lambda self, x: (self.gate(x), real(self.shared_experts, x))[1]))
    elif name == 'no_linear_attn':
        stack.enter_context(patch.object(L.Glm5NextLinearAttention, '__call__',
                                         lambda self, x, mask=None, cache=None: x))
    elif name == 'no_gdn':
        def fake_gdn(q, k, v, a, b, A_log, dt_bias, cache=None, **kw):
            return v, None
        stack.enter_context(patch.object(L, 'gated_delta_update', fake_gdn))
    elif name == 'no_sparse_attn':
        stack.enter_context(patch.object(L.Glm5NextAttention, '__call__',
                                         lambda self, x, *a, **k: (x, None)))
    elif name == 'no_indexer':
        def fake_indexer(self, x, *a, **k):
            return mx.full((x.shape[0], x.shape[1], self.index_topk + self.index_kpool - 1), 0, dtype=mx.int32)
        stack.enter_context(patch.object(L.Glm5NextIndexer, '__call__', fake_indexer))
    elif name == 'sa_no_attend':
        def attend(self, q, latent, new_latent, topk, kv_cache, projected_cache, last_only=False):
            b, h, l, _ = q.shape
            return mx.zeros((b, l, h * self.v_head_dim), dtype=q.dtype), topk
        stack.enter_context(patch.object(L.Glm5NextAttention, '_attend', attend))
    elif name == 'sa_no_sdpa':
        stack.enter_context(patch.object(L, 'scaled_dot_product_attention', lambda q, k, v, **kw: q))
    elif name == 'no_hc':
        def apply_branch(self, x, norm, branch, *args, **kwargs):
            out = branch(norm(x[:, :, 0]), *args, **kwargs)
            if isinstance(out, tuple):
                return (x + out[0][:, :, None], *out[1:])
            return x + out[:, :, None]
        stack.enter_context(patch.object(hcmod.HyperConnection, 'apply_branch', apply_branch))
    elif name == 'no_head':
        stack.enter_context(patch.object(model, 'lm_head', lambda h: h))
    elif name.startswith('fast'):
        import backbone_fast
        options = dict(ASYNC_EVERY=4)
        for part in name.split('_')[1:]:
            if part.startswith('a'):
                options['ASYNC_EVERY'] = int(part[1:])
        for key, value in options.items():
            stack.enter_context(patch.object(backbone_fast, key, value))
        stack.enter_context(backbone_fast.patched(model))
    else:
        raise ValueError(name)
    return stack


GPU_ONLY = False
NO_INNER_ASYNC = False
BLOCK = 1


def prefill(model, ids, chunk=8):
    cache = model.make_cache()
    for i in range(0, len(ids), chunk):
        logits = model(mx.array([ids[i:i + chunk]]), cache=cache).logits
        mx.eval(logits, [c.state for c in cache])
    return cache, logits


def run_variant(model, name, ids, forced, steps, repeat=1):
    with variant_patches(name, model):
        cache, logits = prefill(model, ids)
        build, total, evalt, async_launch, gpu_only = [], [], [], [], []
        big_a = mx.random.normal((4096, 4096)); mx.eval(big_a)
        def big():
            y = big_a
            for _ in range(12):
                y = (y @ big_a) * 1e-3
            return y
        t = time.perf_counter(); mx.eval(big()); 
        big_ms = []
        for _ in range(5):
            t = time.perf_counter(); mx.eval(big()); big_ms.append(time.perf_counter() - t)
        big_time = statistics.median(big_ms)
        ops = None
        outs = []
        for i in range(steps):
            for layer in model.layers:
                if isinstance(layer.mlp, SharedOnly):
                    layer.mlp.trace = []
            t0 = time.perf_counter()
            if NO_INNER_ASYNC:
                real_async = mx.async_eval
                mx.async_eval = lambda *a, **k: None
                try:
                    logits = model(mx.array([forced[i * BLOCK:(i + 1) * BLOCK]]), cache=cache).logits
                finally:
                    mx.async_eval = real_async
            else:
                logits = model(mx.array([forced[i * BLOCK:(i + 1) * BLOCK]]), cache=cache).logits
            extra = [v for layer in model.layers if isinstance(layer.mlp, SharedOnly) for t in layer.mlp.trace for v in t]
            t1 = time.perf_counter()
            if i == 0:
                ops = count_ops([logits] + [c.state for c in cache])
            t1b = time.perf_counter()
            mx.async_eval(logits, [c.state for c in cache], extra)
            t2 = time.perf_counter()
            mx.eval(logits)
            t3 = time.perf_counter()
            outs.append(logits[0].astype(mx.float32))
            if GPU_ONLY:
                # Queue a long independent job first so host encoding of the step overlaps it; the extra
                # wall time over the job alone is the step's GPU execution time.
                real_async = mx.async_eval
                mx.async_eval = lambda *a, **k: None  # keep in-graph async evals from running before the job
                try:
                    logits = model(mx.array([forced[i * BLOCK:(i + 1) * BLOCK]]), cache=cache).logits
                finally:
                    mx.async_eval = real_async
                extra = [v for layer in model.layers if isinstance(layer.mlp, SharedOnly) for t in layer.mlp.trace for v in t]
                b = big()
                t4 = time.perf_counter()
                mx.async_eval(b)
                mx.async_eval(logits, [c.state for c in cache], extra)
                mx.eval(b, logits)
                gpu_only.append(time.perf_counter() - t4 - big_time)
            build.append(t1 - t0)
            async_launch.append(t2 - t1b)
            evalt.append(t3 - t1b)
            total.append(t3 - t1b + t1 - t0)
        med = lambda v: round(1000 * statistics.median(v[4:]), 3)
        return dict(variant=name, build_ms=med(build), launch_ms=med(async_launch), gpu_wall_ms=med(evalt),
                    step_ms=med(total), gpu_only_ms=med(gpu_only) if gpu_only else None, ops_total=sum(ops.values()), ops=dict(ops.most_common())), outs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--backbone', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=40)
    parser.add_argument('--variants', default='base')
    parser.add_argument('--rounds', type=int, default=1)
    parser.add_argument('--compare', default=None, help='variant to compare logits against base')
    parser.add_argument('--block', type=int, default=1)
    parser.add_argument('--compare-blocks', default=None)
    parser.add_argument('--sync-profile', action='store_true')
    parser.add_argument('--no-inner-async', action='store_true')
    parser.add_argument('--gpu-only', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    global GPU_ONLY, BLOCK, NO_INNER_ASYNC
    NO_INNER_ASYNC = args.no_inner_async
    GPU_ONLY = args.gpu_only
    BLOCK = args.block
    mx.set_cache_limit(256 * 1024**2)
    mx.set_memory_limit(24 * 1024**3)
    model = load_prepared(args.model, args.backbone, lambda *a: None)
    install(model, SharedOnly)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    template = (Path(__file__).resolve().parents[1] / 'spark/flash/thinking_off.jinja').read_text()
    ids = tokenizer.apply_chat_template([{'role': 'user', 'content': 'Explain how a hash map works, in detail.'}],
                                        tokenize=True, return_dict=False, add_generation_prompt=True,
                                        chat_template=template)
    # Forced tokens: greedy continuation from the baseline backbone-only model.
    cache, logits = prefill(model, ids)
    forced = []
    for _ in range(args.steps * args.block):
        tok = int(logits[0, -1].argmax().item())
        forced.append(tok)
        logits = model(mx.array([[tok]]), cache=cache).logits
    del cache
    if args.sync_profile:
        out = [sync_profile(model, ids, forced, args.steps, v) for v in args.variants.split(',')]
        for o in out:
            print(json.dumps(o), flush=True)
        args.output.write_text(json.dumps(out, indent=1))
        return
    results = []
    names = args.variants.split(',')
    for r in range(args.rounds):
        for name in names:
            res, outs = run_variant(model, name, ids, forced, args.steps)
            res['round'] = r
            print(json.dumps({k: v for k, v in res.items() if k != 'ops'}), flush=True)
            results.append(res)
    comparison = None
    if args.compare_blocks:
        # Reference: the original code one token at a time. Compare the original and the fast path on
        # blocks of BLOCK tokens against it, position by position.
        block = BLOCK
        BLOCK = 1
        _, ref = run_variant(model, 'base', ids, forced, args.steps * block)
        ref = mx.concatenate(ref, axis=0)
        BLOCK = block
        comparison = []
        for name in ('base', args.compare_blocks):
            _, new = run_variant(model, name, ids, forced, args.steps)
            new = mx.concatenate(new, axis=0)
            diff = mx.abs(new - ref).max(axis=-1)
            top1 = (new.argmax(-1) == ref.argmax(-1)).astype(mx.float32).mean().item()
            comparison.append(dict(variant=name, block=block, against='original code, one token per step',
                                   max_abs_logit_diff=diff.max().item(), mean_max_abs=diff.mean().item(),
                                   top1_agreement=top1, positions=int(ref.shape[0])))
            print(json.dumps(comparison[-1]), flush=True)
    if args.compare:
        _, ref = run_variant(model, 'base', ids, forced, args.steps)
        _, new = run_variant(model, args.compare, ids, forced, args.steps)
        diffs = [float(mx.abs(a - b).max().item()) for a, b in zip(ref, new)]
        top1 = [x for a, b in zip(ref, new) for x in (a.argmax(-1) == b.argmax(-1)).tolist()]
        comparison = dict(variant=args.compare, max_abs_logit_diff=max(diffs), mean_max_abs=statistics.mean(diffs),
                          top1_agreement=sum(top1) / len(top1), positions=len(top1), block=BLOCK)
        print(json.dumps(comparison), flush=True)
    args.output.write_text(json.dumps(dict(prompt_tokens=len(ids), results=results, comparison=comparison), indent=1))



def sync_profile(model, ids, forced, steps, variant='base'):
    """Per-component GPU time: evaluate inputs, then time the component's own evaluation.

    Adds a synchronization per component, so totals exceed the pipelined step; the fixed cost of a
    synchronization is measured on a trivial op and subtracted."""
    from collections import defaultdict
    from mlx.utils import tree_flatten
    totals = defaultdict(float)
    counts = defaultdict(int)
    stack_names = []

    def arrays(tree):
        return [v for _, v in tree_flatten(tree) if isinstance(v, mx.array)]

    def wrap(cls, attr, name):
        original = getattr(cls, attr)

        def wrapper(self, *a, **k):
            mx.eval(arrays((a, k)))
            if stack_names:
                # nested: the parent's timer still runs; record the child separately
                pass
            stack_names.append(name)
            t = time.perf_counter()
            out = original(self, *a, **k)
            mx.eval(arrays(out))
            dt = time.perf_counter() - t
            stack_names.pop()
            totals[name] += dt
            counts[name] += 1
            if stack_names:
                totals[stack_names[-1] + '(-children)'] -= dt
            return out
        return patch.object(cls, attr, wrapper)

    import backbone_fast
    with ExitStack() as stack:
        stack.enter_context(variant_patches(variant, model))
        cache, _ = prefill(model, ids)
        stack.enter_context(wrap(hcmod.HyperConnection, '__call__', 'hc_collapse'))
        stack.enter_context(wrap(nn.RMSNorm, '__call__', 'rmsnorm'))
        stack.enter_context(wrap(language.Glm5NextLinearAttention, '__call__', 'linear_attn'))
        stack.enter_context(wrap(language.Glm5NextAttention, '__call__', 'sparse_attn'))
        stack.enter_context(wrap(language.Glm5NextIndexer, '__call__', 'indexer'))
        stack.enter_context(wrap(SharedOnly, '__call__', 'shared_expert+gate'))
        stack.enter_context(wrap(language.Glm5NextMLP, '__call__', 'mlp'))
        orig_expand = hcmod.hc_expand

        def timed_expand(*a, **k):
            mx.eval(arrays(a))
            t = time.perf_counter()
            out = orig_expand(*a, **k)
            mx.eval(out)
            totals['hc_expand'] += time.perf_counter() - t
            counts['hc_expand'] += 1
            return out
        stack.enter_context(patch.object(hcmod, 'hc_expand', timed_expand))
        x = mx.ones((1, 4096)); mx.eval(x)
        sync = []
        for _ in range(200):
            t = time.perf_counter(); mx.eval(x + 1); sync.append(time.perf_counter() - t)
        sync_cost = statistics.median(sync)
        walls = []
        for i in range(steps):
            t = time.perf_counter()
            logits = model(mx.array([forced[i * BLOCK:(i + 1) * BLOCK]]), cache=cache).logits
            mx.eval(logits)
            walls.append(time.perf_counter() - t)
    result = {name: round(1000 * (totals[name] - counts[name] * sync_cost * (0 if name.endswith(')') else 1)) / steps, 3)
              for name in totals}
    result = dict(sorted(result.items(), key=lambda kv: -kv[1]))
    return dict(variant=variant, block=BLOCK, sync_cost_us=round(sync_cost * 1e6, 1),
                calls_per_step={k: v // steps for k, v in counts.items()},
                ms_per_step=result, synced_wall_ms=round(1000 * statistics.median(walls), 2))


if __name__ == '__main__':
    main()
