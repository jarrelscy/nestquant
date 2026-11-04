import argparse
import json
import statistics
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import torch

from experts import ExpertStore, NestQuantExperts
from metrics import MemoryGuard, snapshot


EXPERTS = (0, 7, 31, 63, 127, 191, 255, 287)


def timing(values):
    return dict(median_seconds=statistics.median(values), minimum_seconds=min(values),
                maximum_seconds=max(values), samples_seconds=values)


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runs', type=int, default=20)
    args = parser.parse_args()
    if args.output.exists() or not 3 <= args.runs <= 100:
        parser.error('Use a fresh output and between three and one hundred runs')
    torch.set_num_threads(8)
    mx.set_cache_limit(256 * 1024**2)
    store = ExpertStore(args.model, cache_gib=4, resident_base=True,
                        async_refinements=True, base_cache_gib=3)
    state = SimpleNamespace(phase='decode', routing={}, wanted=np.zeros((42, 288), dtype=bool),
                            decode_ready_experts=0)
    modules = [NestQuantExperts(layer, store, state) for layer in range(3, 45)]
    array = np.random.default_rng(42).normal(size=(1, 4096)).astype(np.float16).astype(np.float32)
    hidden = torch.from_numpy(array).to('mps')
    ids = torch.tensor([EXPERTS], device='mps')
    weights = torch.full((1, 8), 1 / 8, device='mps')
    entered, release = threading.Event(), threading.Event()
    read_windows = []
    original = store.prefetch.read

    guard = MemoryGuard(10)

    def held_read(layer, expert):
        entered.set()
        if not release.wait(30):
            raise TimeoutError('The diagnostic did not release the worker')
        started = time.perf_counter()
        try:
            return original(layer, expert)
        finally:
            read_windows.append((started, time.perf_counter()))

    def cycle():
        guard.check(force=True)
        torch.mps.synchronize()
        started = time.perf_counter()
        outputs = [module(hidden, ids, weights) for module in modules]
        torch.mps.synchronize()
        stopped = time.perf_counter()
        if not all(torch.isfinite(value).all().item() for value in outputs):
            raise FloatingPointError('Nonfinite expert output')
        return stopped - started, outputs, (started, stopped)

    store.memory_check = guard.check
    report = dict(status='running', scope='Fixed input and eight experts in each of 42 real layers; excludes backbone, predictor and token generation',
                  async_refinements=True, mtp=False, layers=list(range(3, 45)), experts=list(EXPERTS),
                  source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  source_dirty=bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for layer in range(3, 45):
            for expert in EXPERTS:
                store.get(layer, expert, 2)
        if store.metrics['base_evictions']:
            raise RuntimeError('The fixed working set must fit without base eviction')
        cycle()
        cycle()
        cold_samples = []
        for _ in range(args.runs):
            elapsed, cold_output, _ = cycle()
            cold_samples.append(elapsed)
        report['resident_base_cycle'] = timing(cold_samples)
        store.prefetch.read = held_read
        state.wanted[:, EXPERTS] = True
        budgets = {layer: len(EXPERTS) for layer in range(3, 45)}
        store.check_hot_capacity(budgets)
        store.begin_decode(state.wanted, np.ones((42, 288)), budgets)
        if not entered.wait(5):
            raise TimeoutError('The asynchronous worker did not start')
        before_ready = state.decode_ready_experts
        before_fallbacks = store.metrics['async_fallbacks']
        elapsed, blocked_output, _ = cycle()
        if release.is_set() or not store.prefetch.pending or any(f.done() for f, _ in store.prefetch.pending.values()):
            raise RuntimeError('The read was not blocked throughout the decode cycle')
        if state.decode_ready_experts != before_ready or store.metrics['async_fallbacks'] - before_fallbacks != 336:
            raise RuntimeError('The blocked cycle did not use all 336 base fallbacks')
        for actual, expected in zip(blocked_output, cold_output):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        report['blocked_reader_cycle'] = dict(seconds=elapsed, base_fallbacks=336,
                                              worker_finished=False, output_matches_base=True)
        release.set()
        windows = []
        delivery_start = time.perf_counter()
        deadline = delivery_start + 120
        while len(store.cache) < 336:
            if time.perf_counter() > deadline:
                raise TimeoutError('Real refinement delivery exceeded 120 seconds')
            _, _, window = cycle()
            windows.append(window)
        store.finish_decode()
        delivery_seconds = time.perf_counter() - delivery_start
        overlap = sum(max(0, min(re, ce) - max(rs, cs)) for rs, re in read_windows for cs, ce in windows)
        if overlap <= 0:
            raise RuntimeError('No real read overlapped an expert execution interval')
        before = store.metrics.copy()
        hot_samples = []
        for _ in range(args.runs):
            ready = state.decode_ready_experts
            elapsed, _, _ = cycle()
            if state.decode_ready_experts - ready != 336:
                raise RuntimeError('The hot cycle did not use all resident refinements')
            hot_samples.append(elapsed)
        if store.metrics['base_misses'] != before['base_misses'] or store.metrics['refinement_misses'] != before['refinement_misses']:
            raise RuntimeError('A resident timing sample performed a synchronous weight read')
        report.update(status='passed', resident_hot_cycle=timing(hot_samples),
                      real_delivery=dict(seconds=delivery_seconds, reads=len(read_windows),
                                         read_seconds=sum(end - start for start, end in read_windows),
                                         overlap_with_expert_intervals_seconds=overlap, expert_cycles=len(windows),
                                         cycle_timing=timing([end - start for start, end in windows]),
                                         wall_seconds_per_cycle=delivery_seconds / len(windows),
                                         wall_cycle_to_base_median_ratio=delivery_seconds / len(windows) / statistics.median(cold_samples),
                                         expert_interval_fraction=sum(end - start for start, end in windows) / delivery_seconds,
                                         reader_duty_fraction=sum(end - start for start, end in read_windows) / delivery_seconds),
                      metrics=store.metrics.copy(), base_bytes=store.base_bytes, refinement_bytes=store.cache_bytes,
                      staging_reserved_bytes=store.staging_budget, peak_staging_reserved_bytes=store.prefetch.peak_reserved_bytes,
                      resources=snapshot(), limitations=['Not end-to-end tokens per second', 'Fixed routing and inputs',
                                                         'Overlap is measured between wall-clock intervals, not GPU occupancy',
                                                         'No MTP, attention, predictor or autoregressive state',
                                                         'Delivery includes first-use hot kernel compilation and main-thread installation',
                                                         'Allocator cache is 256 MiB rather than the full runtime 1 GiB',
                                                         'Delivery precision changes, so timing ratios do not isolate contention'])
        if store.cache_bytes + store.staging_budget > store.cache_budget or store.base_bytes > store.base_cache_limit:
            raise RuntimeError('A tensor cache exceeded its budget')
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        release.set()
        store.close()
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
