import argparse
import json
from pathlib import Path

import mlx.core as mx

from compact_base import verify
from decoder import projection_matvec, projection_matvec_banked
from experts import ExpertStore
from metrics import MemoryGuard
from microbench_banked import EXPERTS
from native_assets import read_bank
from packed_expert_bank import pack


def compare(source, compact, store, layer, check):
    original = read_bank(source, layer, 'mlx-nocache')
    candidate = read_bank(compact, layer, 'mlx-nocache')
    compared = verify(original, candidate)
    weights = [store.get(layer, expert, 4) for expert in EXPERTS]
    check()
    route = mx.array(EXPERTS, dtype=mx.uint32)
    reports = []
    for name, (planes, meta) in original.items():
        packed, geometry = candidate[name]
        hot = pack([{key: value for key, value in expert[name][0].items()
                     if key in ('p4', 'word', 'suh4', 'svh4', 'lrU4')} for expert in weights])
        rank = planes['lrV'].shape[1]
        if 'lrU4' not in hot:
            hot['lrU4'] = mx.zeros((len(EXPERTS), rank, meta['n']), dtype=mx.float16)
        else:
            hot['lrU4'] = mx.pad(hot['lrU4'], ((0, 0), (0, rank - hot['lrU4'].shape[1]), (0, 0)))
        inputs = mx.random.normal((len(EXPERTS), meta['k'])).astype(mx.float16).astype(mx.float32)
        for mode in ('base', 'mixed', 'refined', 'cold'):
            selected = [index if mode == 'refined' or (mode == 'mixed' and index % 2) else 0xffffffff
                        for index in range(len(EXPERTS))]
            extra = {} if mode == 'base' else dict(refinements=hot, refined_route=mx.array(selected, dtype=mx.uint32))
            expected = projection_matvec_banked(planes, meta, inputs, route, 2, **extra)
            actual = projection_matvec_banked(packed, geometry, inputs, route, 2, **extra)
            if not mx.isfinite(actual).all().item() or not mx.array_equal(expected, actual).item():
                raise RuntimeError(f'Compact bank changed layer {layer} {name} {mode}')
            serial = mx.stack([projection_matvec(*expert[name], inputs[index],
                                                2 if selected[index] == 0xffffffff else 4)
                               for index, expert in enumerate(weights)])
            relative = (mx.linalg.norm(actual - serial) / mx.linalg.norm(serial)).item()
            if not relative < 0.001:
                raise RuntimeError(f'Compact bank reference mismatch: {relative}')
            reports.append(dict(projection=name, mode=mode, identical=True, serial_relative_rms=relative))
            check()
    return dict(layer=layer, compared_bytes=compared, projections=reports)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--compact-archive', required=True, type=Path)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output path')
    mx.random.seed(42)
    mx.set_cache_limit(256 * 1024**2)
    guard = MemoryGuard(6)
    store = ExpertStore(args.model, cache_gib=0)
    store.memory_check = lambda: guard.check(force=True)
    records = []
    try:
        for layer in (3, 44):
            record = compare(args.archive, args.compact_archive, store, layer, store.memory_check)
            records.append(record)
            print(json.dumps(record), flush=True)
            store.close()
            mx.clear_cache()
    finally:
        store.close()
    with args.output.open('x') as handle:
        json.dump(dict(status='passed', checks=records), handle, indent=2)
        handle.write('\n')


if __name__ == '__main__':
    main()
