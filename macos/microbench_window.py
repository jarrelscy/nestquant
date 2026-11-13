"""Compare banked decoder loops on random packed planes with the GLM-5.3-Flash expert geometry."""

import argparse
import json
import statistics
import time

import mlx.core as mx

from decoder import projection_matvec_banked

GEOMETRY = {
    'gate': dict(k=4096, n=2048, tk=32, tn=128, shard_axis='n', base_K=1.5, res_rule=dict(kind='uniform', K=2.5)),
    'up': dict(k=4096, n=2048, tk=32, tn=128, shard_axis='n', base_K=1.5, res_rule=dict(kind='uniform', K=2.5)),
    'down': dict(k=2048, n=4096, tk=16, tn=256, shard_axis='k', base_K=1.5, res_rule=dict(kind='uniform', K=2.8125)),
}


def words(shape):
    # Format constraint: 1 <= Mb <= 255 and Mb + N <= 257, so the folded code stays within 1024.
    mb = mx.random.randint(1, 256, shape)
    n = mx.floor(mx.random.uniform(shape=shape) * (258 - mb)).astype(mx.int32)
    return (mb + (n << 8)).astype(mx.int32)


def make_layer(depth, hot_rows, var_bits):
    layer = {}
    for name, meta in GEOMETRY.items():
        meta = dict(meta, var_bits=var_bits)
        units = meta['tk'] * meta['tn']
        base = {
            'base': mx.random.randint(0, 256, (depth, units * int(meta['base_K'] * 256))).astype(mx.uint8),
            'var': mx.random.randint(0, 256 if var_bits == 1 else 2, (depth, units * var_bits)).astype(mx.uint8),
            'p4': mx.zeros((1, 1), dtype=mx.uint8), 'word': mx.zeros((1, 1), dtype=mx.int32),
            'suh2': (1 + 0.1 * mx.random.normal((depth, meta['k']))).astype(mx.float16),
            'svh2': (1 + 0.1 * mx.random.normal((depth, meta['n']))).astype(mx.float16),
        }
        refinements = {
            'p4': mx.random.randint(0, 256, (hot_rows, units * int(meta['res_rule']['K'] * 256))).astype(mx.uint8),
            'word': words((hot_rows, units)),
            'suh4': (1 + 0.1 * mx.random.normal((hot_rows, meta['k']))).astype(mx.float16),
            'svh4': (1 + 0.1 * mx.random.normal((hot_rows, meta['n']))).astype(mx.float16),
        }
        layer[name] = (base, refinements, meta)
    mx.eval(layer)
    return layer


def expert_mlp(layer, x, route, refined_route, mode):
    def project(name, value):
        base, refinements, meta = layer[name]
        mixed = refined_route is not None
        return projection_matvec_banked(base, meta, value, route, 2,
                                        refinements=refinements if mixed else None,
                                        refined_route=refined_route, loop_mode=mode)

    gate = mx.minimum(project('gate', x), 10.0)
    up = mx.clip(project('up', x), -10.0, 10.0)
    return project('down', gate * mx.sigmoid(gate) * up)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layers', type=int, default=6)
    parser.add_argument('--depth', type=int, default=288)
    parser.add_argument('--rows', type=int, default=8)
    parser.add_argument('--hot-rows', type=int, default=16)
    parser.add_argument('--var-bits', type=int, choices=(1, 8), default=1)
    parser.add_argument('--samples', type=int, default=20)
    parser.add_argument('--modes', default='inline,split,window')
    args = parser.parse_args()
    mx.random.seed(0)
    layers = [make_layer(args.depth, args.hot_rows, args.var_bits) for _ in range(args.layers)]
    x = mx.random.normal((args.rows, 4096)).astype(mx.float16).astype(mx.float32)
    routes = [mx.array(sorted(set(int(v) for v in mx.random.permutation(args.depth)[:args.rows].tolist())),
                       dtype=mx.uint32) for _ in layers]
    cold = 0xffffffff
    mixed_routes = [mx.array([i % args.hot_rows if i % 2 == 0 else cold for i in range(args.rows)], dtype=mx.uint32)
                    for _ in layers]
    modes = args.modes.split(',')
    result = dict(args=vars(args), precision={}, timing={})
    for precision, refined in (('base', None), ('mixed', mixed_routes), ('refined', [
            mx.arange(args.rows, dtype=mx.uint32) % args.hot_rows for _ in layers])):
        outputs = {}
        for mode in modes:
            outputs[mode] = mx.stack([expert_mlp(layer, x, route, refined[i] if refined else None, mode)
                                      for i, (layer, route) in enumerate(zip(layers, routes))])
            mx.eval(outputs[mode])
        reference = outputs[modes[0]]
        result['precision'][precision] = {
            mode: float(mx.sqrt(mx.mean(mx.square(value - reference)) / mx.mean(mx.square(reference))).item())
            for mode, value in outputs.items()}
        result['timing'][precision] = {}
        for mode in modes:
            def token():
                return [expert_mlp(layer, x, route, refined[i] if refined else None, mode)
                        for i, (layer, route) in enumerate(zip(layers, routes))]
            for _ in range(3):
                mx.eval(token())
            samples = []
            for _ in range(args.samples):
                start = time.perf_counter()
                mx.eval(token())
                samples.append((time.perf_counter() - start) / args.layers)
            result['timing'][precision][mode] = dict(median_ms_per_layer=1000 * statistics.median(samples),
                                                     min_ms_per_layer=1000 * min(samples))
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
