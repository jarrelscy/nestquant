"""Routed-expert kernel microbenchmark on real resident base planes (one or two layers, not the full runtime).

  alu     : GPU instruction throughput probe (int add/mul, fp32/fp16 fma), to convert kernel time into cycles/weight.
  layer   : expert_mlp_fused per MoE layer, chained like the model (each layer waits for the previous one), for
            decode-1 / verify-2 / verify-8 routes at base, mixed and all-refined precision. Refinements are random
            planes with the real geometry (the timing of the decode does not depend on the values).
  stages  : the same, stopping after each of the five launches, to attribute time to prep/window/post kernels.

--variants a,b runs A/B interleaved (round-robin, --rounds) over decoder settings given in VARIANTS or --extra JSON;
every variant is first compared with the first one on the same inputs (bitwise or max abs / rel RMS).
Run inside the GPU lock:  /tmp/nq_model_lock.sh <owner> python bench_expert_kernels.py layer ...
"""
import argparse
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
import mlx.core as mx

import decoder

COLD = 0xffffffff
BANKS = Path.home() / 'Models/GLM/resident-banks-v1'
VARIANTS = {'default': {}, 'default2': {}}


def words(shape):
    mb = mx.random.randint(1, 256, shape)
    n = mx.floor(mx.random.uniform(shape=shape) * (258 - mb)).astype(mx.int32)
    return (mb + (n << 8)).astype(mx.int32)


def load_layer(layer, hot, random_base=False):
    if random_base:
        from microbench_window import make_layer
        raw = make_layer(288, hot, 8)
        bank = {name: (dict(raw[name][0]), raw[name][2]) for name in raw}
        for name, (planes, meta) in bank.items():
            planes['lrV'] = (0.05 * mx.random.normal((288, 2, meta['k']))).astype(mx.float16)
            planes['lrU2'] = (0.05 * mx.random.normal((288, 2, meta['n']))).astype(mx.float16)
    else:
        import native_assets
        bank = native_assets.read_bank(BANKS, layer)
    refinements = {}
    for name, (planes, meta) in bank.items():
        units = meta['tk'] * meta['tn']
        rank = planes['lrV'].shape[1] if 'lrV' in planes else 0
        refinements[name] = {
            'p4': mx.random.randint(0, 256, (hot, units * int(meta['res_rule']['K'] * 256))).astype(mx.uint8),
            'word': words((hot, units)),
            'suh4': (1 + 0.1 * mx.random.normal((hot, meta['k']))).astype(mx.float16),
            'svh4': (1 + 0.1 * mx.random.normal((hot, meta['n']))).astype(mx.float16),
        }
        if rank:
            refinements[name]['lrU4'] = (0.05 * mx.random.normal((hot, rank, meta['n']))).astype(mx.float16)
    mx.eval(bank, refinements)
    return bank, refinements


def make_route(tokens, distinct, rng):
    pool = rng.sample(range(288), distinct)
    while True:
        ids = [e for _ in range(tokens) for e in rng.sample(pool, 8)]
        if len(set(ids)) == distinct:
            return ids


SCENARIOS = (('decode-1', 1, 8), ('verify-2', 2, 15), ('verify-8', 8, 47))


def scenarios(hot, refined_fraction, keep):
    rng = random.Random(3)
    out = []
    for label, tokens, distinct in SCENARIOS:
        ids = make_route(tokens, distinct, rng)
        uniques = list(dict.fromkeys(ids))
        chosen = set(rng.sample(uniques, round(len(uniques) * refined_fraction)))
        slot = {e: i % hot for i, e in enumerate(uniques)}
        route = mx.array(ids, dtype=mx.uint32)
        mixed = mx.array([slot[e] if e in chosen else COLD for e in ids], dtype=mx.uint32)
        full = mx.array([slot[e] for e in ids], dtype=mx.uint32)
        x = mx.random.normal((tokens, 4096)).astype(mx.float16).astype(mx.float32)
        mx.eval(route, mixed, full, x)
        for precision, refined in (('base', None), ('mixed', mixed), ('refined', full)):
            if not keep or f'{label}/{precision}' in keep:
                out.append((f'{label}/{precision}', x, route, refined, distinct))
    return out


def mlp(layer, x, route, refined):
    bank, refinements = layer
    return decoder.expert_mlp_fused(bank, x, route, 8, refinements=refinements if refined is not None else None,
                                    refined_route=refined)


def chained(layers, depth, x, route, refined):
    def run():
        value, outs = x, []
        for i in range(depth):
            y = mlp(layers[i % len(layers)], value, route, refined)
            outs.append(y)
            value = x + 0 * y.reshape(-1)[0]
        return outs
    return run


def apply(settings):
    for key, value in settings.items():
        if key.startswith('ENV_'):
            os.environ[key[4:]] = str(value)
        elif key == 'SRC_RE':
            continue
        else:
            setattr(decoder, key, tuple(value) if isinstance(value, list) else value)


_KERNELS = {}
CURRENT = ['']


def _per_variant(name):
    # Kernel factories cached per variant (their settings are module globals), so variants can alternate sample
    # by sample.
    raw = getattr(decoder, name).__wrapped__

    def cached(*a, **k):
        key = (name, CURRENT[0], a, tuple(sorted(k.items())))
        if key not in _KERNELS:
            _KERNELS[key] = raw(*a, **k)
        return _KERNELS[key]
    cached.cache_clear = lambda: None
    setattr(decoder, name, cached)


for _name in ('_window_kernel', '_prep_pair_kernel', '_prep_kernel', '_post_pair_kernel', '_post_kernel'):
    _per_variant(_name)
_RAW_METAL_KERNEL = mx.fast.metal_kernel


def _ablated_kernel(**kw):
    # Timing-only ablations: a variant's SRC_RE [[regex, replacement], ...] edits the window kernel source.
    edits = VARIANTS.get(CURRENT[0], {}).get('SRC_RE')
    if edits and kw['name'].startswith('nq_matvec'):
        import re
        for pattern, replacement in edits:
            kw['source'], hits = re.subn(pattern, replacement, kw['source'])
            assert hits, pattern
        kw['name'] += '_abl_' + CURRENT[0]
    return _RAW_METAL_KERNEL(**kw)


mx.fast.metal_kernel = _ablated_kernel


def snapshot(names):
    return {k: getattr(decoder, k) for k in names if not k.startswith('ENV_')}


def cmd_layer(args):
    extra = open(args.extra[1:]).read() if args.extra.startswith('@') else args.extra
    for name, settings in json.loads(extra).items():
        VARIANTS[name] = settings
    names = args.variants.split(',')
    keys = {k for n in names for k in VARIANTS[n] if k != 'SRC_RE'}
    original = snapshot(keys)
    mx.random.seed(0)
    layers = [load_layer(l, args.hot, args.random_base) for l in args.layers]
    cases = scenarios(args.hot, args.refined_fraction, set(args.scenarios.split(',')) if args.scenarios else None)
    reference, diffs = {}, {n: {} for n in names}
    for name in names:
        CURRENT[0] = name
        apply({**original, **VARIANTS[name]})
        for key, x, route, refined, _ in cases:
            out = mx.stack([mlp(layer, x, route, refined) for layer in layers])
            mx.eval(out)
            reference.setdefault(key, out)
            ref = reference[key]
            if mx.array_equal(out, ref, equal_nan=True).item():
                diffs[name][key] = 'bitwise'
            else:
                d = mx.abs(out - ref)
                d = mx.where(mx.isnan(d), 0, d)
                rel = mx.sqrt(mx.mean(mx.square(d)) / mx.mean(mx.square(mx.where(mx.isnan(ref), 0, ref))))
                nan_mismatch = mx.sum(mx.isnan(out) != mx.isnan(ref)).item()
                diffs[name][key] = f'max {mx.max(d).item():.3g} relrms {rel.item():.2g} nanmismatch {nan_mismatch}'
    samples = {n: {} for n in names}
    order = list(names)
    for _ in range(args.rounds):
        for key, x, route, refined, _ in cases:
            for name in names:
                CURRENT[0] = name
                apply({**original, **VARIANTS[name]})
                mx.eval(chained(layers, args.depth, x, route, refined)())
            for _ in range(args.samples):
                random.shuffle(order)
                for name in order:   # alternate variants sample by sample (thermal drift hits all alike)
                    CURRENT[0] = name
                    apply({**original, **VARIANTS[name]})
                    outs = chained(layers, args.depth, x, route, refined)()
                    start = time.perf_counter()
                    mx.eval(outs)
                    samples[name].setdefault(key, []).append((time.perf_counter() - start) / args.depth * 1e3)
    apply(original)
    print('ms per layer (median of rounds x samples; spread = IQR/median)')
    print('scenario'.ljust(18) + ''.join(n[:20].rjust(24) for n in names))
    for key, *_ in cases:
        base = statistics.median(samples[names[0]][key])
        cells = []
        for n in names:
            s = samples[n][key]
            m = statistics.median(s)
            q = statistics.quantiles(s, n=4)
            spread = (q[2] - q[0]) / m * 100
            cell = f'{m:.3f}±{spread:.0f}%' + ('' if n == names[0] else f' {100 * (m / base - 1):+.1f}%')
            cells.append(cell.rjust(24))
        print(key.ljust(18) + ''.join(cells))
    for n in names[1:]:
        print(n, json.dumps(diffs[n]))
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(dict(args=vars(args), samples=samples, diffs=diffs), f)


def cmd_alu(args):
    ops = {
        'iadd32': ('uint', 'v{i} = v{i} + (v{i} >> 3) ;', 2),
        'iand_shift': ('uint', 'v{i} = (v{i} >> 3) & 0x0F0F0F0Fu ^ v{i};', 3),
        'imul32': ('uint', 'v{i} = v{i} * 0x83DCD12Du;', 1),
        'imad32': ('uint', 'v{i} = v{i} * 0x83DCD12Du + 7u;', 1),
        'ffma32': ('float', 'v{i} = fma(v{i}, 0.999f, 0.001f);', 1),
        'hfma16x2': ('half2', 'v{i} = fma(v{i}, half2(0.999h), half2(0.001h));', 2),
        'cvt_ffma': ('float', 'v{i} = fma(float(half(v{i})), 0.999f, 0.001f);', 1),
    }
    threads, iters, chains = 40 * 1024 * 8, 4096, 8
    for name, (ty, body, flops) in ops.items():
        decl = ''.join(f'{ty} v{i} = {ty}(thread_position_in_grid.x + {i});\n' for i in range(chains))
        loop = ''.join(body.format(i=i) + '\n' for i in range(chains))
        acc = ' + '.join(f'float(v{i}{".x" if ty == "half2" else ""})' for i in range(chains))
        src = decl + f'for (uint it = 0; it < ITERS; ++it) {{\n{loop}}}\nout[thread_position_in_grid.x] = {acc};\n'
        kernel = mx.fast.metal_kernel(name=f'alu_{name}', input_names=['dummy'], output_names=['out'], source=src)
        call = lambda: kernel(inputs=[mx.zeros((1,))], template=[('ITERS', iters)], grid=(threads, 1, 1),
                              threadgroup=(256, 1, 1), output_shapes=[(threads,)], output_dtypes=[mx.float32])[0]
        for _ in range(3):
            mx.eval(call())
        times = []
        for _ in range(9):
            start = time.perf_counter()
            mx.eval([call() for _ in range(4)])
            times.append((time.perf_counter() - start) / 4)
        t = statistics.median(times)
        instr = threads * iters * chains * flops
        print(f'{name:12s} {instr / t / 1e12:6.2f} T instr/s  ({t * 1e3:.2f} ms)')


def cmd_stages(args):
    """Chained time of expert_mlp_fused truncated after each launch (prefix differences = per-stage latency)."""
    mx.random.seed(0)
    layers = [load_layer(l, args.hot, args.random_base) for l in args.layers]
    cases = scenarios(args.hot, args.refined_fraction, set(args.scenarios.split(',')) if args.scenarios else None)
    original = decoder.expert_mlp_fused

    def truncated(stop):
        # Each named kernel factory returns a kernel; after `stop` launches, return the last output.
        state = {'count': 0}

        class Stop(Exception):
            pass

        def wrap(factory):
            def make(*a, **k):
                kernel = factory(*a, **k)

                def call(*ca, **ck):
                    out = kernel(*ca, **ck)
                    state['count'] += 1
                    if state['count'] >= stop:
                        state['out'] = out
                        raise Stop
                    return out
                return call
            return make

        def run(*a, **k):
            saved = {n: getattr(decoder, n) for n in ('_prep_pair_kernel', '_window_kernel', '_post_pair_kernel', '_post_kernel')}
            for n, f in saved.items():
                setattr(decoder, n, wrap(f))
            state['count'] = 0
            try:
                return original(*a, **k)
            except Stop:
                return state['out'][0]
            finally:
                for n, f in saved.items():
                    setattr(decoder, n, f)
        return run

    labels = ['prep_pair', 'window_gate_up', 'post_pair', 'window_down', 'post_down']
    for key, x, route, refined, distinct in cases:
        times = {}
        for _ in range(args.rounds):
            for stop in range(1, 6):
                decoder.expert_mlp_fused = truncated(stop)
                try:
                    fn = chained(layers, args.depth, x, route, refined)
                    mx.eval(fn())
                    for _ in range(args.samples):
                        start = time.perf_counter()
                        mx.eval(fn())
                        times.setdefault(stop, []).append((time.perf_counter() - start) / args.depth * 1e3)
                finally:
                    decoder.expert_mlp_fused = original
        med = [statistics.median(times[s]) for s in range(1, 6)]
        parts = [med[0]] + [med[i] - med[i - 1] for i in range(1, 5)]
        weights = distinct * 3 * 4096 * 2048
        print(f'{key:18s} total {med[-1]:.3f} ms/layer  ' + '  '.join(f'{l} {p:.3f}' for l, p in zip(labels, parts))
              + f'  | {weights / med[-1] / 1e9:.0f} G weights/s overall')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['alu', 'layer', 'stages'])
    parser.add_argument('--layers', type=int, nargs='+', default=[10, 11])
    parser.add_argument('--random-base', action='store_true')
    parser.add_argument('--hot', type=int, default=64)
    parser.add_argument('--depth', type=int, default=12, help='chained MoE layers per sample')
    parser.add_argument('--samples', type=int, default=10)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--refined-fraction', type=float, default=0.5)
    parser.add_argument('--scenarios', default='')
    parser.add_argument('--variants', default='default,default2')
    parser.add_argument('--extra', default='{}')
    parser.add_argument('--output')
    args = parser.parse_args()
    {'alu': cmd_alu, 'layer': cmd_layer, 'stages': cmd_stages}[args.command](args)


if __name__ == '__main__':
    main()
