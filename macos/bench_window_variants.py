"""Time routed-expert MoE layers (gate, up, SwiGLU, down via projection_fused) under decoder variants.

Random packed planes with the GLM-5.3-Flash geometry. Each variant is checked bit-for-bit against the
default kernel on the same inputs. Run inside the GPU lock:
  /tmp/nq_model_lock.sh <owner> python bench_window_variants.py --variants default,sum_add,...
"""
import argparse
import json
import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import mlx.core as mx

import decoder
from microbench_window import make_layer

COLD = 0xffffffff
DEFAULTS = dict(SPLIT_MIN_ROWS=9, SPLIT_PRECISION=False, XV_FLOAT=False, SUM_MODE='mul', HASH_MODE='mul32', MAD_MODE='mul32', SPECIALIZED_MEMBERS=(4, 1),
                LANE_POSITIONS=16, GROUP_MAXR_OVERRIDE=None)
VARIANTS = {
    'default': {},
    'default2': {},
    'sum_add': dict(SUM_MODE='add'),
    'hash24': dict(HASH_MODE='mul24'),
    'mad24': dict(MAD_MODE='mad24'),
    'int_all': dict(SUM_MODE='add', HASH_MODE='mul24', MAD_MODE='mad24'),
    'sum_add_mad24': dict(SUM_MODE='add', MAD_MODE='mad24'),
    'ref2': dict(SPECIALIZED_MEMBERS=(4, 2)),
    'ref4': dict(SPECIALIZED_MEMBERS=(4, 4)),
    'maxr2': dict(GROUP_MAXR_OVERRIDE=2),
    'maxr8': dict(GROUP_MAXR_OVERRIDE=8),
    'xvf': dict(XV_FLOAT=True),
    'split': dict(SPLIT_PRECISION=True),
    'fused_split_all': dict(FUSED=True, SPLIT_PRECISION=True, SPLIT_MIN_ROWS=1),
    'fused_split': dict(FUSED=True, SPLIT_PRECISION=True),
    'fused': dict(FUSED=True),
    'fused_sum_add': dict(FUSED=True, SUM_MODE='add'),
}
FUSED = [False]


def apply(settings):
    settings = dict(settings)
    FUSED[0] = settings.pop('FUSED', False)
    for key, value in {**DEFAULTS, **settings}.items():
        setattr(decoder, key, value)
    decoder._window_kernel.cache_clear()


def mlp(layer, x, ids, refined):
    if FUSED[0]:
        tokens = x.reshape(-1, 8, x.shape[-1])[:, 0]
        planes = {name: (layer[name][0], layer[name][2]) for name in ('gate', 'up', 'down')}
        refinements = {name: layer[name][1] for name in ('gate', 'up', 'down')} if refined is not None else None
        return decoder.expert_mlp_fused(planes, tokens, ids, 8, refinements=refinements, refined_route=refined)

    def project(name, value):
        base, refinements, meta = layer[name]
        options = dict(refinements=refinements, refined_route=refined) if refined is not None else {}
        return decoder.projection_fused(base, meta, value, ids, **options)
    gate = mx.minimum(project('gate', x), 10.0)
    up = mx.clip(project('up', x), -10.0, 10.0)
    return project('down', gate * mx.sigmoid(gate) * up)


def timed(fn, n):
    for _ in range(4):
        mx.eval(fn())
    samples = []
    for _ in range(n):
        start = time.perf_counter()
        mx.eval(fn())
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variants', default='default,sum_add,hash24,mad24,int_all,ref2,ref4')
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--samples', type=int, default=25)
    parser.add_argument('--refined-fraction', type=float, default=0.4)
    parser.add_argument('--output')
    args = parser.parse_args()
    mx.random.seed(0)
    random.seed(3)
    layers = [make_layer(288, 64, 1) for _ in range(args.layers)]
    scenarios = []
    for label, tokens, uniq in (('decode-1', 1, 8), ('verify-2', 2, 13), ('verify-4', 4, 23), ('verify-8', 8, 37)):
        pool = random.sample(range(288), uniq)
        ids_list = []
        for _ in range(tokens):
            ids_list += random.sample(pool, 8)
        ids = mx.array(ids_list, dtype=mx.uint32)
        uniques = list(dict.fromkeys(ids_list))
        hot = set(uniques[:round(len(uniques) * args.refined_fraction)])
        slot = {e: i % 64 for i, e in enumerate(uniques)}
        refined = mx.array([slot[e] if e in hot else COLD for e in ids_list], dtype=mx.uint32)
        all_refined = mx.array([slot[e] for e in ids_list], dtype=mx.uint32)
        x = mx.repeat(mx.random.normal((tokens, 4096)).astype(mx.float16).astype(mx.float32), 8, axis=0)
        mx.eval(x, ids, refined, all_refined)
        for precision, route in (('base', None), ('mixed', refined), ('refined', all_refined)):
            scenarios.append((label, precision, x, ids, route))
    reference = {}
    results = []
    for name in args.variants.split(','):
        apply(VARIANTS[name])
        row = dict(variant=name, ms_per_layer={}, max_abs_diff={})
        try:
            run_variant(args, name, row, scenarios, layers, reference)
        except Exception as error:   # e.g. a variant that does not compile
            row['error'] = f'{type(error).__name__}: {error}'[:400]
        results.append(row)
        print(json.dumps(row), flush=True)
    apply({})
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=1)


def run_variant(args, name, row, scenarios, layers, reference):
        for label, precision, x, ids, route in scenarios:
            key = f'{label}/{precision}'
            out = mx.stack([mlp(layer, x, ids, route) for layer in layers])
            mx.eval(out)
            if name == args.variants.split(',')[0]:
                reference[key] = out
            diff = mx.abs(out - reference[key])
            row['max_abs_diff'][key] = float(mx.max(mx.where(mx.isnan(diff), 0, diff)).item())
            ms = timed(lambda: [mlp(layer, x, ids, route) for layer in layers], args.samples) / args.layers * 1e3
            row['ms_per_layer'][key] = round(ms, 4)


if __name__ == '__main__':
    main()
