"""Standalone CPU/SSD benchmark: shard RefinementReader vs ExpertMajorRefinementReader (no MLX operations).

Each configuration reads a disjoint random set of (layer, expert) refinements with 1/4/8 threads and reports
per-read latency and aggregate GiB/s, optionally with a concurrent pure-Python thread that keeps the GIL busy
(a stand-in for the decode loop), and optionally including the install copy into bank-shaped host planes
(PackedRefinementBank.write). With the page cache enabled, a second pass re-reads the same keys (RAM hits).

  python macos/bench_refinement_reader.py --model <checkpoint> --archive <dir> --output /tmp/bench.json
"""

import argparse
import itertools
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import numpy as np

import refinement_archive as ra


def spinner(stop, counter):
    while not stop.is_set():
        total = 0
        for i in range(2000):
            total += i * i
        counter[0] += 1


def run(make_reader, keys, threads, contention, install, repeat=1, direct=False):
    reader = make_reader()
    if hasattr(reader, '_handle'):  # open shard handles and parse headers outside the timed region
        for layer in {layer for layer, _ in keys}:
            reader._manifest(layer)
            for shard in range(8):
                reader._handle(layer, shard)
    hosts = {}
    from packed_expert_bank import PackedRefinementBank
    if install:
        for layer in {layer for layer, _ in keys}:
            plan = ra.plan_layer(ARGS.model, ARGS.base, layer)
            hosts[layer] = {}
            for item in plan['fields']:
                hosts[layer].setdefault(item['projection'], {})[item['field']] = np.zeros(
                    (threads, *item['shape']), dtype=ra.DTYPES[item['dtype']])
    local = threading.local()
    slots = itertools.count()
    slot_lock = threading.Lock()

    def one(key):
        if install and not hasattr(local, 'slot'):
            with slot_lock:
                local.slot = next(slots) % threads
        started = time.perf_counter()
        copy = 0.0
        if direct:
            count, _ = reader.read_into_slot(*key, SimpleNamespace(host=hosts[key[0]]), local.slot)
            return time.perf_counter() - started, count, copy
        payload, count, _ = reader.read(*key)
        if install:
            copy_started = time.perf_counter()
            PackedRefinementBank.write(SimpleNamespace(host=hosts[key[0]]), local.slot, payload)
            copy = time.perf_counter() - copy_started
        del payload
        return time.perf_counter() - started, count, copy

    results = []
    stop, counter = threading.Event(), [0]
    spin = threading.Thread(target=spinner, args=(stop, counter), daemon=True) if contention else None
    try:
        if spin:
            spin.start()
        for _ in range(repeat):
            started = time.perf_counter()
            with ThreadPoolExecutor(max_workers=threads) as executor:
                values = list(executor.map(one, keys))
            wall = time.perf_counter() - started
            latencies = np.array([v[0] for v in values]) * 1e3
            total = sum(v[1] for v in values)
            results.append(dict(reads=len(keys), gib=total / 2**30, wall_s=wall, gib_per_s=total / 2**30 / wall,
                                latency_ms_mean=float(latencies.mean()), latency_ms_p50=float(np.median(latencies)),
                                latency_ms_p95=float(np.percentile(latencies, 95)),
                                install_copy_ms_mean=float(np.mean([v[2] for v in values]) * 1e3) if install else None))
    finally:
        stop.set()
        if spin:
            spin.join()
        reader.close()
    for entry in results:
        entry['spinner_iterations_per_s'] = counter[0] / sum(r['wall_s'] for r in results) if contention else None
    return results


def main():
    global ARGS
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--base', type=Path, default=Path('~/Models/GLM/resident-banks-v1').expanduser())
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--reads', type=int, default=96, help='Reads per configuration')
    parser.add_argument('--threads', default='1,4,8')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=1)
    ARGS = parser.parse_args()
    from experts import RefinementReader
    from native_refinements import LockedRefinementReader

    rng = np.random.default_rng(ARGS.seed)
    pool = [(int(l), int(e)) for l, e in rng.permutation([(l, e) for l in ra.LAYERS for e in range(288)])]
    position = 0

    def take(n):
        nonlocal position
        keys = pool[position:position + n]
        position += n
        if len(keys) < n:
            raise RuntimeError('Ran out of distinct keys')
        return keys

    readers = {
        'shards': lambda: LockedRefinementReader(ARGS.model),
        'archive_nocache': lambda: ra.ExpertMajorRefinementReader(ARGS.archive, root=ARGS.model, cache=False),
        'archive_nocache_nora': lambda: ra.ExpertMajorRefinementReader(ARGS.archive, root=ARGS.model, cache=False,
                                                                      readahead=False),
        'archive_nocache_nora_direct': lambda: ra.ExpertMajorRefinementReader(ARGS.archive, root=ARGS.model,
                                                                             cache=False, readahead=False),
        'archive_cache': lambda: ra.ExpertMajorRefinementReader(ARGS.archive, root=ARGS.model, cache=True,
                                                               readahead=False),
    }
    report = []
    for threads in (int(t) for t in ARGS.threads.split(',')):
        for contention in (False, True):
            for name, make in readers.items():
                repeat = 2 if name == 'archive_cache' else 1
                results = run(make, take(ARGS.reads), threads, contention, True, repeat, direct=name.endswith('_direct'))
                for pass_index, result in enumerate(results):
                    entry = dict(reader=name, threads=threads, gil_contention=contention,
                                 pass_=('cold' if pass_index == 0 else 'warm_reread'), **result)
                    report.append(entry)
                    print(json.dumps(entry), flush=True)
    ARGS.output.write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
