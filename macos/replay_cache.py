import argparse
from collections import OrderedDict
import json
from pathlib import Path
import time


PROMPT = 'Write a Python function that merges overlapping intervals. Explain its algorithm, edge cases, and time complexity. Include examples.'


def simulate(accesses, limit_bytes):
    cache = OrderedDict()
    used = 0
    counts = {phase: dict(accesses=0, misses=0, miss_bytes=0, evictions=0)
              for phase in ('prefill', 'decode')}
    for phase, layer, expert, size in accesses:
        stats = counts[phase]
        stats['accesses'] += 1
        key = layer, expert
        if key in cache:
            if cache[key] != size:
                raise ValueError('An expert changed size during the replay')
            cache.move_to_end(key)
            continue
        if not 0 < size <= limit_bytes:
            raise ValueError('An expert cannot fit in the replay cache')
        stats['misses'] += 1
        stats['miss_bytes'] += size
        while used + size > limit_bytes:
            _, removed = cache.popitem(last=False)
            used -= removed
            stats['evictions'] += 1
        cache[key] = size
        used += size
    return counts


def capture(model, tokens, cache_gib, chunk, async_base, profile_crossings):
    import mlx.core as mx
    import torch
    from generate import Runtime
    from metrics import snapshot
    from predictor import predictor_memory

    torch.set_num_threads(8)
    mx.set_cache_limit(1024**3)
    startup = time.perf_counter()
    runtime = Runtime(model, 4, hot_fraction=0.02, resident_base=True,
                      async_refinements=True, base_cache_gib=cache_gib, async_base=async_base)
    startup = time.perf_counter() - startup
    trace = []
    store = runtime.store
    store.profile_crossings = profile_crossings
    original = store.get
    in_get = False

    def traced(layer, expert, level):
        nonlocal in_get
        if in_get:
            return original(layer, expert, level)
        in_get = True
        try:
            result = original(layer, expert, level)
            trace.append((runtime.phase, layer, expert, store.base_sizes[layer, expert]))
            return result
        finally:
            in_get = False

    def delta(before):
        return {key: value - before[key] for key, value in store.metrics.items()}

    store.get = traced
    try:
        ids = runtime.tokenizer.apply_chat_template(
            [dict(role='user', content=PROMPT)], tokenize=True, return_dict=False,
            add_generation_prompt=True, chat_template=runtime.thinking_off_template)
        before = store.metrics.copy()
        measurements = {'loaded': snapshot()}
        start = time.perf_counter()
        logits = runtime.prefill(ids, chunk)
        timings = {'prefill': time.perf_counter() - start}
        prefill = delta(before)
        measurements['after_prefill'] = snapshot()
        before = store.metrics.copy()
        greedy = [int(logits.argmax())]
        start = time.perf_counter()
        for token in tokens:
            greedy.append(int(runtime.step(token).argmax()))
        timings['decode'] = time.perf_counter() - start
        runtime.store.finish_decode()
        runtime.memory_guard.check(force=True)
        decode = delta(before)
        measurements['after_decode'] = snapshot()
        expected = {'prefill': prefill, 'decode': decode}
        observed = simulate(trace, int(cache_gib * 1024**3))
        for phase in expected:
            if (observed[phase]['misses'], observed[phase]['evictions']) != (
                    expected[phase]['base_misses'], expected[phase]['base_evictions']):
                raise RuntimeError(f'The cache replay differs from the observed {phase} counts')
        return dict(startup_seconds=startup, prompt_tokens=len(ids), timings=timings, async_base=async_base,
                    profile_crossings=profile_crossings,
                    teacher_forced_ids=tokens, greedy_next_ids=greedy, metrics=expected,
                    resources=measurements, final_base_bytes=store.base_bytes,
                    final_refinement_bytes=store.cache_bytes, predictor_seconds=runtime.predictor_seconds,
                    predictor_memory=predictor_memory(runtime.predictor),
                    base_staging_peak=store.base_prefetch.peak_reserved_bytes if store.base_prefetch else 0,
                    predictor=runtime.predictor_metrics.report(), base_accesses=trace,
                    capacities={size: simulate(trace, size * 1024**3)
                                for size in (32, 36, 40, 44, 48, 56)})
    finally:
        runtime.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--fixed-json', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base-cache-gib', type=int, choices=(32,), default=32)
    parser.add_argument('--async-base', action='store_true')
    parser.add_argument('--profile-crossings', action='store_true')
    parser.add_argument('--prefill-chunk-size', type=int, choices=(8, 16, 32), default=8)
    parser.add_argument('--tokens', type=int, choices=(32, 64, 128), default=64)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output file')
    tokens = json.loads(args.fixed_json.read_text())['generated_ids'][:args.tokens]
    if len(tokens) != args.tokens or any(type(token) is not int or token < 0 for token in tokens):
        parser.error('The input must contain enough valid generated token IDs')
    result = capture(args.model, tokens, args.base_cache_gib, args.prefill_chunk_size, args.async_base, args.profile_crossings)
    with args.output.open('x') as output:
        json.dump(result, output, indent=2)
        output.write('\n')
