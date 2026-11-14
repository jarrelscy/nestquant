"""Profile plain and DFlash2 decode at several context lengths on the full runtime (server configuration).

For each context length: a fresh batched prefill of repository source text plus an editing instruction, then
  1. plain pipelined decode (as native_generate.generate) for --plain-tokens tokens,
  2. DFlash2 adaptive decode (native_dflash.dflash_decode, the server path) for --tokens tokens, with a
     per-round host timeline,
  3. isolated costs with the cache rolled back after each call: verify forward per block size, drafter per
     block size, and verify with components removed (expert matvec, routed experts, linear attention,
     sparse attention).
Run inside the GPU lock: /tmp/nq_model_lock.sh <owner> python profile_dflash.py --output out.json
"""

import argparse
from contextlib import ExitStack
import gc
import json
from pathlib import Path
import statistics
import time
from unittest.mock import patch

import mlx.core as mx
import numpy as np
import torch

HOME = Path.home()
INSTRUCTION = ('Rewrite the last complete function shown above with full type hints and a concise docstring. '
               'Reply with only the code.')


def med(values):
    return round(1000 * statistics.median(values), 3) if values else None


def context_ids(tokenizer, template, files, target):
    body = []
    for path in files:
        body.append(f'### {path.name}\n```python\n{path.read_text()}\n```\n')
    text = '\n'.join(body)
    ids = tokenizer.encode(text, add_special_tokens=False)
    overhead = len(tokenizer.apply_chat_template([{'role': 'user', 'content': INSTRUCTION}], tokenize=True,
                                                 return_dict=False, add_generation_prompt=True,
                                                 chat_template=template))
    keep = max(target - overhead - 4, 16)
    if keep > len(ids):
        raise ValueError(f'Context source has only {len(ids)} tokens')
    content = tokenizer.decode(ids[:keep]) + '\n\n' + INSTRUCTION
    return tokenizer.apply_chat_template([{'role': 'user', 'content': content}], tokenize=True, return_dict=False,
                                         add_generation_prompt=True, chat_template=template)


def plain_decode(ng, model, cache, hot, guard, token, steps, wanted, threaded=False):
    """native_generate.generate's pipelined loop on an existing cache, also capturing drafter hidden rows.
    threaded: the previous token's predictor commit and this token's plan run on a thread while the forward is
    built (the build waits on the GPU)."""
    hot.start_pipelined_decode()
    pending = None
    pool = None
    if threaded:
        from concurrent.futures import ThreadPoolExecutor
        pool = ThreadPoolExecutor(max_workers=1)
    times, tokens, rows = [], [], []
    gc.disable()
    for _ in range(steps):
        tokens.append(token)
        start = time.perf_counter()
        hot.publish()
        job = None
        if pool is not None:
            def host(pending=pending, token=token):
                if pending is not None:
                    hot.commit(*pending)
                hot.plan(token)
            job = pool.submit(host)
        out = model(mx.array([[token]]), cache=cache, capture_layer_ids=wanted) if wanted else model(mx.array([[token]]), cache=cache)
        logits = out.logits
        row = mx.concatenate(out.hidden_states, axis=-1) if wanted else logits
        trace = hot.take_trace()
        mx.async_eval(logits, row, [entry.state for entry in cache],
                      [value for layer in trace.values() for value in layer[1:3]])
        if job is not None:
            job.result()
        else:
            if pending is not None:
                hot.commit(*pending)
            hot.plan(token)
        pending = ([token], 0, trace)
        token = ng.greedy(logits)
        times.append(time.perf_counter() - start)
        rows.append(row)
        guard.check()
    hot.commit(*pending)
    if pool is not None:
        pool.shutdown()
    gc.enable()
    gc.collect()
    return token, tokens, times, (mx.concatenate(rows, axis=1) if wanted else None)


def take_checkpoint(cache, hot):
    from mlx_vlm.models.cache import KVCache
    from native_prefix import arrays, clone, leaves, predictor_snapshot
    parts = []
    for leaf in leaves(cache):
        parts.append(('kv', leaf.offset) if type(leaf) is KVCache else ('snapshot', clone(leaf.prefix_cache_snapshot())))
    mx.eval(arrays([part for kind, part in parts if kind == 'snapshot']))
    predictor, _ = predictor_snapshot(hot)
    return dict(parts=parts, predictor=predictor, position=hot.position)


def restore_checkpoint(cache, hot, checkpoint):
    """dflash_server's _restore_checkpoint followed by _resume_prefill."""
    from native_prefix import arrays, clone, leaves, restore_predictor
    for leaf, (kind, part) in zip(leaves(cache), checkpoint['parts'], strict=True):
        if kind == 'kv':
            leaf.offset = part
        else:
            leaf.prefix_cache_restore(clone(part))
    mx.eval(arrays([leaf.state for leaf in leaves(cache)]))
    restore_predictor(hot, checkpoint['predictor'])
    predictor = hot.predictor
    predictor.ahead, predictor.ahead_started = None, False
    hot.position = checkpoint['position']
    hot.phase = 'prefill'
    hot.prefill_bank = None
    hot.prefill_layer = None
    hot.chunk_index = 0
    hot.traces.clear()
    if hot.position != predictor.t:
        raise RuntimeError('Predictor clock differs from the committed token count')


class Timeline:
    """Host wall time spent in each piece of the mlx-vlm DFlash round loop."""

    def __init__(self, dl, drafter):
        self.dl, self.drafter = dl, drafter
        self.totals = {}

    def wrap(self, owner, name, label):
        original = getattr(owner, name)

        def timed(*a, **k):
            start = time.perf_counter()
            try:
                return original(*a, **k)
            finally:
                self.totals[label] = self.totals.get(label, 0.0) + time.perf_counter() - start
        return patch.object(owner, name, timed)

    def patches(self):
        stack = ExitStack()
        stack.enter_context(self.wrap(self.dl, '_speculative_walk', 'walk_wait'))
        stack.enter_context(self.wrap(self.dl, 'commit_speculative_round', 'commit_cache'))
        stack.enter_context(self.wrap(self.dl, '_dflash_verify_greedy', 'verify_build'))
        if hasattr(self.drafter, 'draft_block_greedy'):
            stack.enter_context(self.wrap(self.drafter, 'draft_block_greedy', 'draft_build'))
        else:
            stack.enter_context(self.wrap(self.drafter, 'draft_block', 'draft_build'))
        if hasattr(self.drafter, 'prepare_target_hidden'):
            stack.enter_context(self.wrap(self.drafter, 'prepare_target_hidden', 'prepare_hidden'))
        return stack


def isolated_costs(args, nd, ng, dl, model, drafter, cache, hot, ctx, tokens, blocks, variants):
    from mlx_vlm.speculative.cache_state import abort_speculative_round
    wanted = list(drafter.config.target_layer_ids)
    lm = model.language_model if hasattr(model, 'language_model') else model
    sampler = lambda value: value.argmax(axis=-1)
    hot.start_pipelined_decode()
    out = {}

    def verify(bs, repeat):
        build, total = [], []
        for r in range(repeat + 2):
            inputs = mx.array([tokens[r:r + bs]], dtype=mx.int32)
            mx.eval(inputs)
            t0 = time.perf_counter()
            captured, state, target = dl._dflash_verify_greedy(lm, inputs, cache, wanted, sampler)
            t1 = time.perf_counter()
            mx.eval(target, captured)
            t2 = time.perf_counter()
            abort_speculative_round(state)
            hot.traces.clear()
            if r >= 2:
                build.append(t1 - t0)
                total.append(t2 - t0)
        return dict(build_ms=med(build), total_ms=med(total))

    if args.interleave:
        # Thermally balanced A/B: every repetition times each variant and block once, in rotation.
        samples = {(v, bs): [] for v in variants for bs in args.variant_blocks}
        for r in range(args.interleave):
            for variant in (variants if r % 2 == 0 else variants[::-1]):
                with variant_patches(variant, ng):
                    for bs in args.variant_blocks:
                        inputs = mx.array([tokens[r % 8:r % 8 + bs]], dtype=mx.int32)
                        mx.eval(inputs)
                        t0 = time.perf_counter()
                        captured, state, target = dl._dflash_verify_greedy(lm, inputs, cache, wanted, sampler)
                        mx.eval(target, captured)
                        samples[(variant, bs)].append(time.perf_counter() - t0)
                        abort_speculative_round(state)
                        hot.traces.clear()
        for variant in variants:
            out[variant] = {bs: dict(total_ms=med(samples[(variant, bs)][2:]),
                                     min_ms=round(1000 * min(samples[(variant, bs)]), 3)) for bs in args.variant_blocks}
            print(json.dumps(dict(event='verify_interleaved', variant=variant, costs=out[variant])), flush=True)
        variants = []
    for variant in variants:
        with variant_patches(variant, ng):
            out[variant] = {bs: verify(bs, args.repeat) for bs in (blocks if variant.startswith('full') else args.variant_blocks)}
        print(json.dumps(dict(event='verify_costs', variant=variant, costs=out[variant])), flush=True)

    draft_cache = drafter.reset(model)
    hidden = ctx
    if hasattr(drafter, 'prepare_target_hidden'):
        hidden = drafter.prepare_target_hidden(hidden)
    fn = getattr(drafter, 'draft_block_greedy', drafter.draft_block)
    mx.eval(fn(tokens[0], hidden, draft_cache, 4, sampler, mx.int32))
    small = ctx[:, -3:]
    if hasattr(drafter, 'prepare_target_hidden'):
        small = drafter.prepare_target_hidden(small)
    mx.eval(small)
    drafts = {}
    for bs in blocks:
        if bs < 2:
            continue
        samples = []
        for r in range(args.repeat + 2):
            t0 = time.perf_counter()
            mx.eval(fn(tokens[r], small, draft_cache, bs, sampler, mx.int32))
            if r >= 2:
                samples.append(time.perf_counter() - t0)
        drafts[bs] = med(samples)
    out['drafter_ms'] = drafts
    print(json.dumps(dict(event='drafter_costs', costs=drafts)), flush=True)
    return out


def variant_patches(name, ng):
    from mlx_vlm.models.glm5_next import language as L
    import decoder
    stack = ExitStack()
    name = name.split('#')[0]          # 'full#2' repeats a variant
    if name == 'full':
        return stack
    if name == 'no_weight_once':
        import backbone_fast
        stack.enter_context(patch.object(backbone_fast, 'WEIGHT_ONCE', False))
        return stack
    if name == 'unfused':
        stack.enter_context(patch.object(ng, 'FUSED_MLP', False))
        return stack
    if name in ('split', 'nosplit', 'unfused_nosplit'):
        import decoder
        stack.enter_context(patch.object(decoder, 'SPLIT_PRECISION', name == 'split'))
        if name == 'unfused_nosplit':
            stack.enter_context(patch.object(ng, 'FUSED_MLP', False))
        return stack
    if name == 'baseline':
        import backbone_fast
        stack.enter_context(patch.object(backbone_fast, 'WEIGHT_ONCE', False))
        stack.enter_context(patch.object(ng, 'FUSED_MLP', False))
        return stack
    if name == 'no_expert_matvec':
        original = decoder.mx.fast.metal_kernel

        def probe(**kw):
            if kw['name'].startswith('nq_matvec_spark') or kw['name'].startswith('nq_matvec_window'):
                kw = dict(kw, name=kw['name'] + '_probe',
                          source=kw['source'].replace('uint group = threadgroup_position_in_grid.x;',
                                                      'partial[0] = 0.0f; return;\n    uint group = threadgroup_position_in_grid.x;', 1))
            return original(**kw)
        decoder._window_kernel.cache_clear()
        stack.enter_context(patch.object(decoder.mx.fast, 'metal_kernel', probe))
        stack.callback(decoder._window_kernel.cache_clear)
    elif name == 'shared_only':
        def shared_only(self, x):
            ids, weights = self.gate(x)
            return self.shared_experts(x)
        stack.enter_context(patch.object(ng.PagedExperts, '__call__', shared_only))
    elif name == 'no_linear_attn':
        stack.enter_context(patch.object(L.Glm5NextLinearAttention, '__call__', lambda self, x, mask=None, cache=None: x))
    elif name == 'no_sparse_attn':
        stack.enter_context(patch.object(L.Glm5NextAttention, '__call__', lambda self, x, *a, **k: (x, None)))
    else:
        raise ValueError(name)
    return stack


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, default=HOME / 'Models/GLM/GLM-5.3-Flash-NestQuant-1.5-4bit')
    parser.add_argument('--archive', type=Path, default=HOME / 'Models/GLM/resident-banks-v1')
    parser.add_argument('--backbone', type=Path, default=HOME / 'Models/GLM/backbone-affine4-e8-v1')
    parser.add_argument('--drafter', type=Path, default=HOME / 'Models/GLM/drafters/dflash2')
    parser.add_argument('--contexts', default='1000,8000,16000,30000')
    parser.add_argument('--tokens', type=int, default=192)
    parser.add_argument('--plain-tokens', type=int, default=48)
    parser.add_argument('--repeat', type=int, default=6)
    parser.add_argument('--blocks', default='1,2,3,4,5,6,8')
    parser.add_argument('--variant-blocks', default='1,4,8')
    parser.add_argument('--variants', default='full,no_expert_matvec,shared_only,no_linear_attn,no_sparse_attn')
    parser.add_argument('--skip-isolated', action='store_true')
    parser.add_argument('--loops', default='mlx,fast,fastbg', help='DFlash round loops to compare from the same prefill')
    parser.add_argument('--plain-modes', default='sync,bg')
    parser.add_argument('--async-every', type=int, default=None)
    parser.add_argument('--memory-limit-gib', type=int, default=96)
    parser.add_argument('--interleave', type=int, default=0)
    parser.add_argument('--freeze-hot', action='store_true', help='No refinement installs during the DFlash runs')
    parser.add_argument('--fixed-block', type=int, default=0, help='Verify this block every round instead of adaptive')
    parser.add_argument('--gc-freeze', action='store_true', help='gc.freeze() after loading')
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.variant_blocks = [int(v) for v in args.variant_blocks.split(',')]

    import native_dflash as nd
    import native_generate as ng
    from metrics import MemoryGuard
    from native_prefill import BatchedPrefill
    from native_refinements import NativeHotRefinements
    from mlx_vlm.speculative import dflash as dl
    from mlx_vlm.speculative.drafters import load_drafter
    from transformers import AutoTokenizer

    nd._install_capture()
    nd._relax_compatibility()
    mx.set_cache_limit(256 * 1024**2)
    mx.set_memory_limit(args.memory_limit_gib * 1024**3)
    if args.async_every is not None:
        import backbone_fast
        backbone_fast.ASYNC_EVERY = args.async_every
    mx.set_wired_limit(92 * 1024**3)
    torch.set_num_threads(4)
    guard = MemoryGuard(98, 0.8)
    guard.allowed_pressure = (1, 2)
    hot = NativeHotRefinements(args.model, 0.15, 128, 16, nd.READERS, 48, 512)
    loaded = time.perf_counter()
    model, banks = ng.load_runtime(args.model, args.backbone, args.archive, [288] * 42, guard, hot, True, 56)
    results = dict(contexts=[])
    try:
        hot.warm(4, lambda: guard.check(force=True))
        hot.draft_rows_base = False
        nd.BATCHED = BatchedPrefill(args.model)
        drafter, _ = load_drafter(str(args.drafter))
        mx.eval(drafter.parameters())
        results['load_seconds'] = time.perf_counter() - loaded
        if args.gc_freeze:
            gc.collect()
            gc.freeze()
        wanted = list(drafter.config.target_layer_ids)
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        template = (Path(ng.__file__).resolve().parents[1] / 'spark/flash/thinking_off.jinja').read_text()
        files = sorted(args.source.glob('*.py'))
        for target in (int(v) for v in args.contexts.split(',')):
            ids = context_ids(tokenizer, template, files, target)
            entry = dict(context_tokens=len(ids))
            entry['mlx_active_gib_before'] = round(mx.get_active_memory() / 2**30, 2)
            cache = model.make_cache()
            hot.reset()
            started = time.perf_counter()
            logits, hidden = nd.capture_prefill(model, drafter, ids, cache, hot, guard)
            mx.eval(hidden)
            entry['prefill_seconds'] = round(time.perf_counter() - started, 2)
            ctx = mx.contiguous(hidden[:, -2047:])
            del hidden
            token = ng.greedy(logits)
            checkpoint = take_checkpoint(cache, hot)
            entry['dflash'] = {}
            generated = []
            frozen = {}
            if args.freeze_hot:
                # No refinement changes during the loop comparison: every run sees the same 4-bit set.
                hot.drain()
                frozen = dict(publish=hot.publish, flip=hot.flip)
                hot.publish = lambda *a, **k: None
                hot.flip = lambda *a, **k: None
            block = args.fixed_block or -8
            drafter.prefer_requested_block_size = bool(args.fixed_block)
            for number, loop in enumerate(args.loops.split(',')):
                restore_checkpoint(cache, hot, checkpoint)
                hot.drain()
                # mlx: mlx-vlm round loop; fast: reordered loop; *bg: background refinement installs.
                # Optional '+' flags: unfused (separate gate/up/down launches), nowo (per-token backbone qmv).
                base_name, *flags = loop.split('+')
                nd.FAST_LOOP = base_name.startswith('fast')
                hot.background_installs = base_name.endswith('bg')
                import backbone_fast
                import decoder as decoder_module
                reads_before = dict(hot.reads)
                timeline = Timeline(dl, drafter)
                with timeline.patches(), patch.object(ng, 'FUSED_MLP', 'unfused' not in flags), \
                        patch.object(backbone_fast, 'WEIGHT_ONCE', 'nowo' not in flags), \
                        patch.object(decoder_module, 'SPLIT_PRECISION',
                                     decoder_module.SPLIT_PRECISION and 'unfused' not in flags):
                    out = nd.dflash_decode(model, drafter, cache, ctx, token, args.tokens, set(), hot, guard, block)
                rounds = max(out['rounds'], 1)
                generated = generated or list(out['generated_ids'])
                key = f'{number}:{loop}'
                entry['dflash'][key] = dict(
                    decode_tps=round(out['decode_tps'], 2), rounds=out['rounds'],
                    mean_accepted=round(out['mean_accepted'], 3), mean_drafted=round(out['mean_drafted'], 3),
                    median_round_ms=round(out['median_round_ms'], 2),
                    mean_round_ms=round(1000 * out['round_seconds_total'] / rounds, 2),
                    adaptive_cost_ms=out['adaptive_cost_ms'], mean_unique_experts=round(out['mean_unique_experts'], 1),
                    ready_call_fraction=round(out['ready_call_fraction'], 3),
                    host_ms_per_round={k: round(1000 * v / rounds, 2) for k, v in timeline.totals.items()},
                    between_ms_per_round={k: round(1000 * v / rounds, 2) for k, v in out['between_seconds'].items()},
                    cleanup_seconds=round(out['cleanup_seconds'], 3), gc_objects=len(gc.get_objects()),
                    publication_seconds=round(out['publication_seconds'], 3), installs=out['refinement_installs'],
                    hook_wait_ms_per_round=round(1000 * out['hook_wait_seconds'] / rounds, 2),
                    read_gib=round((hot.reads['decode_bytes'] - reads_before['decode_bytes']) / 2**30, 3),
                    read_gib_s=round((hot.reads['decode_bytes'] - reads_before['decode_bytes']) / 2**30
                                     / out['decode_seconds'], 3),
                    reader_busy_ratio=round((hot.reads['decode_worker_seconds'] - reads_before['decode_worker_seconds'])
                                            / out['decode_seconds'], 2),
                    tokens_matching_first_loop=sum(a == b for a, b in zip(generated, out['generated_ids'])),
                    text=tokenizer.decode(out['generated_ids'][:60]))
                print(json.dumps(dict(event='dflash', context=len(ids), loop=key, **entry['dflash'][key])), flush=True)
            for name, value in frozen.items():
                delattr(hot, name)
            drafter.prefer_requested_block_size = False
            nd.FAST_LOOP = True
            entry['plain'] = {}
            for mode in args.plain_modes.split(','):
                restore_checkpoint(cache, hot, checkpoint)
                hot.drain()
                hot.background_installs = mode.startswith('bg')
                published = hot.reads['decode_publication_seconds']
                with patch.object(ng, 'FUSED_MLP', 'unfused' not in mode):
                    _, plain_tokens, times, _ = plain_decode(ng, model, cache, hot, guard, token, args.plain_tokens, None,
                                                             threaded='thread' in mode)
                entry['plain'][mode] = dict(tokens=len(times), decode_tps=round(len(times[4:]) / sum(times[4:]), 2),
                                            median_step_ms=med(times[4:]),
                                            publication_seconds=round(hot.reads['decode_publication_seconds'] - published, 3))
                print(json.dumps(dict(event='plain', context=len(ids), mode=mode, **entry['plain'][mode])), flush=True)
            hot.drain()
            hot.background_installs = True
            if not args.skip_isolated:
                generated = generated + plain_tokens
                entry['isolated'] = isolated_costs(args, nd, ng, dl, model, drafter, cache, hot, ctx, generated,
                                                   [int(v) for v in args.blocks.split(',')],
                                                   args.variants.split(','))
            results['contexts'].append(entry)
            args.output.write_text(json.dumps(results, indent=1))
            del cache, ctx
            mx.clear_cache()
            guard.check(force=True)
    finally:
        hot.close()
        for bank in banks:
            bank.close()
        if nd.BATCHED is not None:
            nd.BATCHED.close()
    args.output.write_text(json.dumps(results, indent=1))


if __name__ == '__main__':
    main()
