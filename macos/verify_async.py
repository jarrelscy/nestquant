import argparse
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import torch

from experts import ExpertStore, NestQuantExperts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output file')
    store = ExpertStore(args.model, cache_gib=0.2, resident_base=True, async_refinements=True)
    reference = ExpertStore(args.model, cache_gib=0)
    entered, release = threading.Event(), threading.Event()
    original_read = store.prefetch.read
    def delayed_read(layer, expert):
        entered.set()
        if not release.wait(30):
            raise TimeoutError('Verification did not release the reader')
        return original_read(layer, expert)
    store.prefetch.read = delayed_read
    experts = [0, 7, 287]
    state = SimpleNamespace(phase='decode', routing={}, wanted=np.zeros((42, 288), dtype=bool),
                            decode_ready_experts=0)
    state.wanted[0, experts] = True
    x = np.random.default_rng(42).normal(size=(1, 4096)).astype(np.float16).astype(np.float32)
    ids = torch.tensor([experts])
    weights = torch.tensor([[0.1, 0.2, 0.7]])
    module = NestQuantExperts(3, store, state)
    checks = []
    try:
        for expert in experts:
            store.get(3, expert, 2)
        reads = store.metrics['base_tensor_bytes']
        expected = sum(reference.apply(3, expert, mx.array(x[0]), 2) * float(np.float16(weight))
                       for expert, weight in zip(experts, weights.numpy()[0]))
        mx.eval(expected)
        store.begin_decode(state.wanted, np.ones((42, 288)), {3: 3})
        assert entered.wait(5)
        cold = module(torch.from_numpy(x), ids, weights).numpy()
        np.testing.assert_allclose(cold[0], np.array(expected), rtol=0, atol=1e-4)
        assert state.decode_ready_experts == 0 and store.metrics['async_fallbacks'] == 3
        assert store.metrics['base_tensor_bytes'] == reads
        checks.append('Decode completed with base weights while the SSD worker was blocked')
        release.set()
        deadline = time.monotonic() + 30
        while len(store.cache) < 3:
            if time.monotonic() >= deadline:
                raise TimeoutError('Refinement delivery did not finish')
            store.poll_refinements()
            time.sleep(0.01)
        for expert in experts:
            actual = store.get(3, expert, 4)
            wanted = reference.get(3, expert, 4)
            for projection, (planes, _) in actual.items():
                for field, value in planes.items():
                    np.testing.assert_array_equal(np.array(value), np.array(wanted[projection][0][field]))
        checks.append('Background-loaded tensors exactly match the independent synchronous loader')
        expected = sum(reference.apply(3, expert, mx.array(x[0]), 4) * float(np.float16(weight))
                       for expert, weight in zip(experts, weights.numpy()[0]))
        mx.eval(expected)
        hot = module(torch.from_numpy(x), ids, weights).numpy()
        np.testing.assert_allclose(hot[0], np.array(expected), rtol=0, atol=1e-4)
        assert state.decode_ready_experts == 3
        assert store.cache_bytes + store.staging_budget <= store.cache_budget
        assert store.prefetch.peak_reserved_bytes <= store.staging_budget
        checks.append('Decode uses installed refinements and stays within cache and staging budgets')
        report = dict(status='passed', checks=checks, metrics=store.metrics.copy(),
                      cache_bytes=store.cache_bytes, staging_budget=store.staging_budget,
                      peak_staging_reserved=store.prefetch.peak_reserved_bytes,
                      scope='Three real layer 3 experts; asynchronous fallback and installation, not a full-model benchmark')
    finally:
        release.set()
        store.close()
        reference.close()
    assert not store.handles and not store.reader.handles and not store.prefetch.pending
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
