import argparse
import json
from pathlib import Path
import threading

import mlx.core as mx
import numpy as np

from experts import ExpertStore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output file')
    store = ExpertStore(args.model, cache_gib=0.2, resident_base=True, async_refinements=True, base_cache_gib=0.011)
    reference = ExpertStore(args.model, cache_gib=0)
    x = mx.array(np.random.default_rng(42).normal(size=4096).astype(np.float32))
    checks = []
    entered, release = threading.Event(), threading.Event()
    try:
        for expert in (0, 7, 287, 0):
            for level in (2, 4):
                actual = store.apply(3, expert, x, level)
                expected = reference.apply(3, expert, x, level)
                mx.eval(actual, expected)
                np.testing.assert_array_equal(np.array(actual), np.array(expected))
                assert store.base_bytes <= store.base_cache_limit
                assert store.cache_bytes <= store.cache_limit
                assert all(key[:2] in store.base_cache for key in store.cache)
            checks.append(dict(expert=expert, base_bytes=store.base_bytes, refinement_bytes=store.cache_bytes,
                               resident_experts=sorted(key[1] for key in store.base_cache)))
        assert store.metrics['base_evictions'] == 2
        assert store.metrics['base_misses'] == 4
        _, removed = store.cache.pop((3, 0, 4))
        store.cache_bytes -= removed
        original_read = store.prefetch.read
        def delayed_read(layer, expert):
            entered.set()
            if not release.wait(30):
                raise TimeoutError('Verification did not release the reader')
            return original_read(layer, expert)
        store.prefetch.read = delayed_read
        wanted = np.zeros((42, 288), dtype=bool)
        wanted[0, 0] = True
        store.begin_decode(wanted, np.ones((42, 288)), {3: 1})
        assert entered.wait(5)
        store.get(3, 7, 2)
        store.get(3, 1, 2)
        assert (3, 0) not in store.base_cache
        release.set()
        for future, _ in store.prefetch.pending.values():
            future.result(timeout=30)
        store.poll_refinements()
        assert (3, 0, 4) not in store.cache and not store.prefetch.pending
        assert store.metrics['async_discarded'] == 1
        assert store.base_bytes <= store.base_cache_limit and store.cache_bytes <= store.cache_limit
        checks.append(dict(check='Real async completion discarded after its base was evicted', passed=True))
        report = dict(status='passed', checks=checks, metrics=store.metrics.copy(),
                      base_limit_bytes=store.base_cache_limit, refinement_limit_bytes=store.cache_limit,
                      scope='Real L3 base eviction and reloading; base and 4-bit outputs exactly equal the uncached loader; no full-model quality or speed claim')
    finally:
        release.set()
        store.close()
        reference.close()
    assert not store.base_cache and not store.base_sizes and not store.cache
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
