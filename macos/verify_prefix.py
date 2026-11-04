import argparse
import json
from pathlib import Path
import time

import mlx.core as mx

from checkpoint import validate
from metrics import MemoryGuard
from native_generate import greedy, layer_major_prefill, load_runtime, memory_sample
from native_refinements import NativeHotRefinements
from native_prefix import PrefixCache, arrays, leaves


def compare(reference, actual):
    for a, b in zip(leaves(reference), leaves(actual), strict=True):
        if a.meta_state != b.meta_state:
            raise RuntimeError('Prefix metadata differs')
        for x, y in zip(arrays(a.state), arrays(b.state), strict=True):
            if not mx.array_equal(x, y).item():
                raise RuntimeError(f'{type(a).__name__} arrays differ')


def run(args, guard):
    model_root = args.model
    validate(model_root)
    hot = NativeHotRefinements(model_root, .02, 128) if args.refinements else None
    model, banks = load_runtime(model_root, args.backbone, args.archive, [96] * 42, guard, hot)
    budget = (240 if args.eviction_only else 480) * 1024**2
    prefix = PrefixCache(model, hot, 8, boundary=16, budget=budget)
    common = [40, 451, 2332, 71, 121, 325, 4765, 1218, 79, 88, 600, 471, 7100, 7500, 75, 36]
    cases = [
        ('initial', common + [37, 45, 79], 0),
        ('identical', common + [37, 45, 79], 19),
        ('shared_suffix', common + [40, 82, 7], 16),
        ('one_before_boundary', common[:-1], 0),
        ('on_boundary', common, 16),
        ('equal_length_other', [42] + common[1:] + [37, 45, 79], 0),
        ('evicting_other', [43] + common[1:-1], 0),
        ('old_request_evicted', common + [37, 45, 79], 0),
    ]
    if args.eviction_only:
        cases = [('initial', common[:8], 0), ('distinct', common[8:], 0),
                 ('old_request_evicted', common[:8], 0)]
    report = []
    try:
        for name, ids, expected in cases:
            if hot is not None:
                hot.reset()
            reference = model.make_cache()
            tick = time.monotonic()
            logits, _ = prefix.prefill(layer_major_prefill, model, ids, reference, hot, guard, reuse=False)
            uncached_seconds = time.monotonic() - tick
            predictor = hot.prefill_state_digest() if hot is not None else None
            if hot is not None:
                hot.reset()
            actual = model.make_cache()
            tick = time.monotonic()
            reused, hits = prefix.prefill(layer_major_prefill, model, ids, actual, hot, guard)
            cached_seconds = time.monotonic() - tick
            if hits != expected:
                raise RuntimeError(f'{name}: got {hits} cached tokens, expected {expected}')
            if name == 'old_request_evicted' and prefix.evictions < 2:
                raise RuntimeError('Expected the checkpoint budget to evict both prior entries')
            if not mx.array_equal(logits, reused).item():
                raise RuntimeError(f'{name}: cached logits differ')
            compare(reference, actual)
            if hot is not None and predictor != hot.prefill_state_digest():
                raise RuntimeError(f'{name}: cached predictor differs')
            if hot is None:
                token = greedy(logits)
                next_ref = model(mx.array([[token]]), cache=reference).logits
                next_cached = model(mx.array([[token]]), cache=actual).logits
                if not mx.array_equal(next_ref, next_cached).item():
                    raise RuntimeError(f'{name}: next-token logits differ')
                compare(reference, actual)
            row = dict(case=name, input_tokens=len(ids), cached_tokens=hits, computed_tokens=len(ids)-hits,
                       uncached_seconds=uncached_seconds, cached_seconds=cached_seconds,
                       cache_bytes=prefix.bytes, evictions=prefix.evictions)
            print(json.dumps(row), flush=True)
            report.append(row)
        return dict(status='passed', refinement=args.refinements, results=report, memory=memory_sample())
    finally:
        if hot is not None:
            hot.close()
        for bank in banks:
            bank.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--model', required=True, type=Path)
    p.add_argument('--archive', required=True, type=Path)
    p.add_argument('--backbone', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--refinements', action='store_true')
    p.add_argument('--eviction-only', action='store_true')
    args = p.parse_args()
    if args.output.exists():
        p.error('Use a fresh output file')
    mx.set_cache_limit(256 * 1024**2)
    mx.set_memory_limit(76 * 1024**3)
    result = run(args, MemoryGuard(82))
    with args.output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
