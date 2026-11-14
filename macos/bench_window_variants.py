"""Time routed-expert MoE layers (gate, up, SwiGLU, down via projection_fused) under decoder variants.

Random packed planes with the GLM-5.3-Flash geometry. Each variant is checked bit-for-bit against the
default kernel on the same inputs. Run inside the GPU lock:
  /tmp/nq_model_lock.sh <owner> python bench_window_variants.py --variants fused_old,fused --rounds 4 --chain
--rounds interleaves the variants (cancels thermal drift); --chain makes each layer depend on the previous one,
as in the model (latency), instead of timing independent layers (throughput, which overstates some gains).
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
DEFAULTS = dict(SPLIT_MIN_ROWS=9, SPLIT_PRECISION=False, XV_FLOAT=False, SUM_MODE='mul', HASH_MODE='mul32', MAD_MODE='mul32',
                SPECIALIZED_MEMBERS=(4, 0), LANE_POSITIONS=16, GROUP_MAXR_OVERRIDE=None, REFINED_GROUP_POSITIONS=16,
                WRAP_SHUFFLE=True, PACK_MODE='nomask')
# The kernels before precision-split launches, one-copy grouped refined rows and the shuffled 90-byte ring window.
OLD = dict(SPLIT_PRECISION=False, SPECIALIZED_MEMBERS=(4, 1), REFINED_GROUP_POSITIONS=8, WRAP_SHUFFLE=False,
           PACK_MODE='mask')
VARIANTS = {
    'old': OLD,
    'fused_old': dict(OLD, FUSED=True),
    'default': {},
    'default2': {},
    'sum_add': dict(SUM_MODE='add'),
    'hash24': dict(HASH_MODE='mul24'),
    'mad24': dict(MAD_MODE='mad24'),
    'int_all': dict(SUM_MODE='add', HASH_MODE='mul24', MAD_MODE='mad24'),
    'sum_add_mad24': dict(SUM_MODE='add', MAD_MODE='mad24'),
    'ref2': dict(SPECIALIZED_MEMBERS=(4, 2)),
    'maxr2': dict(GROUP_MAXR_OVERRIDE=2),
    'maxr8': dict(GROUP_MAXR_OVERRIDE=8),
    'xvf': dict(XV_FLOAT=True),
    'fused_split_all': dict(FUSED=True, SPLIT_PRECISION=True, SPLIT_MIN_ROWS=1),
    'fused': dict(FUSED=True),
    'fused_sum_add': dict(FUSED=True, SUM_MODE='add'),
    'fused_ref2': dict(FUSED=True, SPECIALIZED_MEMBERS=(4, 2)),
    'fused_nogroup': dict(FUSED=True, GROUP_MAXR_OVERRIDE=1),
    'fused_split': dict(FUSED=True, SPLIT_PRECISION=True),
    'fused_fallback': dict(FUSED=True, WRAP_SHUFFLE=False),
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


CHAIN = [False]


def layers_fn(layers, x, ids, route):
    """One MoE pass per layer. With --chain each layer's input depends on the previous layer's output (as in the
    model), so the timing is per-layer latency instead of the throughput of independent layers."""
    if not CHAIN[0]:
        return lambda: [mlp(layer, x, ids, route) for layer in layers]

    def run():
        out, value = [], x
        for layer in layers:
            y = mlp(layer, value, ids, route)
            out.append(y)
            value = x + 0 * y[:1, :1]
        return out
    return run


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
    parser.add_argument('--variants', default='fused_old,fused')
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--samples', type=int, default=25)
    parser.add_argument('--refined-fraction', type=float, default=0.4)
    parser.add_argument('--output')
    parser.add_argument('--rounds', type=int, default=1,
                        help='interleave variants round-robin, --samples per round (cancels thermal drift)')
    parser.add_argument('--chain', action='store_true', help='make each layer depend on the previous one')
    parser.add_argument('--extra', default='{}', help='JSON {name: {SETTING: value}} of ad-hoc variants')
    parser.add_argument('--scenarios', default='', help='comma-separated label/precision keys to run (default all)')
    args = parser.parse_args()
    CHAIN[0] = args.chain
    for name, settings in json.loads(args.extra).items():
        VARIANTS[name] = {k: tuple(v) if isinstance(v, list) else v for k, v in settings.items()}
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
    if args.scenarios:
        keep = set(args.scenarios.split(','))
        scenarios = [s for s in scenarios if f'{s[0]}/{s[1]}' in keep]
    if args.rounds > 1:
        return interleaved(args, scenarios, layers)
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


def interleaved(args, scenarios, layers):
    names = args.variants.split(',')
    reference, diffs, samples = {}, {n: {} for n in names}, {n: {} for n in names}
    for name in names:   # compile and check every variant first
        apply(VARIANTS[name])
        for label, precision, x, ids, route in scenarios:
            key = f'{label}/{precision}'
            out = mx.stack([mlp(layer, x, ids, route) for layer in layers])
            mx.eval(out)
            reference.setdefault(key, out)
            same = mx.array_equal(out, reference[key], equal_nan=True).item()
            diff = mx.abs(out - reference[key])
            rel = mx.sqrt(mx.mean(mx.square(mx.where(mx.isnan(diff), 0, diff))) /
                          mx.mean(mx.square(mx.where(mx.isnan(reference[key]), 0, reference[key]))))
            diffs[name][key] = 'bitwise' if same else f'max {mx.max(mx.where(mx.isnan(diff), 0, diff)).item():.3g} relrms {rel.item():.2g}'
    for _ in range(args.rounds):
        for name in names:
            apply(VARIANTS[name])
            for label, precision, x, ids, route in scenarios:
                key = f'{label}/{precision}'
                fn = layers_fn(layers, x, ids, route)
                mx.eval(fn())
                for _ in range(args.samples):
                    start = time.perf_counter()
                    mx.eval(fn())
                    samples[name].setdefault(key, []).append((time.perf_counter() - start) / args.layers * 1e3)
    apply({})
    keys = [f'{s[0]}/{s[1]}' for s in scenarios]
    print('scenario'.ljust(18) + ''.join(n[:16].rjust(17) for n in names))
    for key in keys:
        base = statistics.median(samples[names[0]][key])
        cells = []
        for n in names:
            m = statistics.median(samples[n][key])
            cells.append((f'{m:.3f}' if n == names[0] else f'{m:.3f} {100 * (m / base - 1):+.1f}%').rjust(17))
        print(key.ljust(18) + ''.join(cells))
    for n in names[1:]:
        print(n, json.dumps(diffs[n]))
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(dict(samples=samples, diffs=diffs), f)


if __name__ == '__main__':
    main()
