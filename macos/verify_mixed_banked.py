import argparse
import json
from pathlib import Path

import mlx.core as mx

from decoder import projection_matvec, projection_matvec_banked
from experts import ExpertStore
from microbench_banked import EXPERTS, pack


def compare(store, layer):
    weights = [store.get(layer, expert, 4) for expert in EXPERTS]
    cold_fields = ('base', 'var', 'suh2', 'svh2', 'lrV', 'lrU2')
    hot_fields = ('p4', 'word', 'suh4', 'svh4', 'lrU4')
    hot_indices = (6, 0, 4, 2)
    route = (7, 0, 3, 4, 5, 6, 2, 1)
    cold = 0xffffffff
    hot_route = mx.array([hot_indices.index(i) if i in hot_indices else cold for i in route], dtype=mx.uint32)
    results = {}
    for name, (_, meta) in weights[0].items():
        base = pack([{k: v for k, v in row[name][0].items() if k in cold_fields} for row in weights])
        base.update(p4=mx.zeros((len(weights), 1), dtype=mx.uint8),
                    word=mx.zeros((len(weights), 1), dtype=mx.int32))
        upgraded = pack([{k: v for k, v in weights[i][name][0].items() if k in hot_fields} for i in hot_indices])
        if 'lrV' in base:
            rank = base['lrV'].shape[1]
            if 'lrU4' not in upgraded:
                upgraded['lrU4'] = mx.zeros((len(hot_indices), rank, meta['n']), dtype=mx.float16)
            else:
                upgraded['lrU4'] = mx.pad(upgraded['lrU4'], ((0, 0), (0, rank - upgraded['lrU4'].shape[1]), (0, 0)))
        x = mx.random.normal((len(route), meta['k'])).astype(mx.float16).astype(mx.float32)
        expected = mx.stack([projection_matvec(*weights[i][name], x[j], 4 if i in hot_indices else 2)
                             for j, i in enumerate(route)])
        actual = projection_matvec_banked(base, meta, x, route, 2,
                                           refinements=upgraded, refined_route=hot_route)
        relative = (mx.linalg.norm(actual - expected) / mx.linalg.norm(expected)).item()
        if not relative < 0.001:
            raise RuntimeError(f'Mixed bank mismatch at layer {layer} {name}: {relative}')
        invalid_route = mx.concatenate([mx.array([len(hot_indices)], dtype=mx.uint32), hot_route[1:]])
        invalid = projection_matvec_banked(base, meta, x, route, 2, refinements=upgraded,
                                           refined_route=invalid_route)
        if not mx.isnan(invalid[0]).all().item() or not mx.isfinite(invalid[1:]).all().item():
            raise RuntimeError('Invalid refinement slot did not fail locally')
        all_cold = mx.full((len(route),), cold, dtype=mx.uint32)
        mixed_cold = projection_matvec_banked(base, meta, x, route, 2,
                                              refinements=upgraded, refined_route=all_cold)
        base_only = projection_matvec_banked(base, meta, x, route, 2)
        if not mx.array_equal(mixed_cold, base_only).item():
            raise RuntimeError(f'All-cold mixed projection differs from base at layer {layer} {name}')
        results[name] = dict(mixed_relative_rms=relative, all_cold_identical=True)
    return dict(layer=layer, relative_rms=results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    args = parser.parse_args()
    mx.random.seed(42)
    mx.set_cache_limit(256 * 1024**2)
    store = ExpertStore(args.model, cache_gib=0.5)
    try:
        checks = [compare(store, layer) for layer in (3, 44)]
    finally:
        store.close()
    print(json.dumps(dict(status='passed', checks=checks)))
