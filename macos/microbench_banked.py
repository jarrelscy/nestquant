import argparse
import json
from pathlib import Path
import random
import statistics
import time

import mlx.core as mx

from decoder import projection_matvec, projection_matvec_banked
from experts import ExpertStore
from packed_expert_bank import pack


EXPERTS = (0, 7, 98, 177, 218, 241, 277, 287)


def timed(call):
    for _ in range(5):
        mx.eval(call())
    samples = []
    for _ in range(25):
        start = time.perf_counter()
        mx.eval(call())
        samples.append(time.perf_counter() - start)
    return dict(median_ms=1000 * statistics.median(samples), samples_ms=[1000 * s for s in samples])


def compare(store, layer, level):
    weights = [store.get(layer, expert, level) for expert in EXPERTS]
    bank = {}
    for name, (_, meta) in weights[0].items():
        if any(value[name][1] != meta for value in weights):
            raise RuntimeError(f'Mismatched bank layout for {name}')
        bank[name] = (pack([value[name][0] for value in weights]), meta)
    mx.eval([value for planes, _ in bank.values() for value in planes.values()])
    route = (7, 2, 3, 0, 5, 1, 6, 4)
    rows = mx.random.normal((len(route), 4096)).astype(mx.float16).astype(mx.float32)
    scores = mx.softmax(mx.random.normal((len(route),)))
    mx.eval(rows, scores)

    def activation(gate, up):
        gate = mx.minimum(gate, 10.0)
        return gate * mx.sigmoid(gate) * mx.clip(up, -10.0, 10.0)

    def serial():
        results = []
        for row, index in enumerate(route):
            expert = weights[index]
            gate = projection_matvec(*expert['gate'], rows[row], level)
            up = projection_matvec(*expert['up'], rows[row], level)
            results.append(projection_matvec(*expert['down'], activation(gate, up), level))
        return mx.stack(results)

    def batched(planes, inputs, indices):
        gate = projection_matvec_banked(planes['gate'], bank['gate'][1], inputs, indices, level)
        up = projection_matvec_banked(planes['up'], bank['up'][1], inputs, indices, level)
        return projection_matvec_banked(planes['down'], bank['down'][1], activation(gate, up), indices, level)

    planes = {name: tensors for name, (tensors, _) in bank.items()}
    expected = mx.stack([store.apply(layer, EXPERTS[index], rows[row], level) for row, index in enumerate(route)])
    device_route = mx.array(route, dtype=mx.uint32)
    actual = batched(planes, rows, route)
    on_device = batched(planes, rows, device_route)
    mx.eval(expected, actual, on_device)
    relative_rms = (mx.linalg.norm(actual - expected) / mx.linalg.norm(expected)).item()
    if not relative_rms < 0.001:
        raise RuntimeError(f'Banked expert parity failed at layer {layer}, level {level}: {relative_rms}')
    if not mx.array_equal(actual, on_device).item():
        raise RuntimeError('Device routes changed the banked result')
    invalid = batched(planes, rows, mx.array((*route[:-1], len(EXPERTS)), dtype=mx.uint32))
    if not mx.isfinite(invalid[:-1]).all().item() or not mx.isnan(invalid[-1]).all().item():
        raise RuntimeError('Invalid device route did not fail in its own row')
    compiled = mx.compile(batched)
    start = time.perf_counter()
    fused = compiled(planes, rows, device_route)
    mx.eval(fused)
    compile_first_call_seconds = time.perf_counter() - start
    compiled_rms = (mx.linalg.norm(fused - on_device) / mx.linalg.norm(on_device)).item()
    invalid = compiled(planes, rows, mx.array((*route[:-1], len(EXPERTS)), dtype=mx.uint32))
    if not mx.isfinite(invalid[:-1]).all().item() or not mx.isnan(invalid[-1]).all().item():
        raise RuntimeError('Compiled invalid device route did not fail in its own row')
    altered = {name: dict(values) for name, values in planes.items()}
    altered['gate'][f'suh{level}'] = planes['gate'][f'suh{level}'] * .9
    altered['gate']['var'] = mx.bitwise_xor(planes['gate']['var'], 1)
    changed_rows = rows * .5
    changed = compiled(altered, changed_rows, device_route)
    changed_reference = batched(altered, changed_rows, device_route)
    changed_rms = (mx.linalg.norm(changed - changed_reference) / mx.linalg.norm(changed_reference)).item()
    if not all(value < 1e-5 for value in (compiled_rms, changed_rms)):
        raise RuntimeError(f'Compiled expert parity failed: {compiled_rms}, {changed_rms}')
    return dict(layer=layer, level=level, relative_rms=relative_rms,
                compiled_relative_rms=compiled_rms, changed_input_relative_rms=changed_rms,
                compiled_relative_rms_limit=1e-5, compile_first_call_seconds=compile_first_call_seconds,
                serial=timed(lambda: (serial() * scores[:, None]).sum(axis=0)),
                banked=timed(lambda: (batched(planes, rows, route) * scores[:, None]).sum(axis=0)),
                device=timed(lambda: (batched(planes, rows, device_route) * scores[:, None]).sum(axis=0)),
                compiled=timed(lambda: (compiled(planes, rows, device_route) * scores[:, None]).sum(axis=0)))


def compare_variants(store, layer, guard, variants):
    weights = [store.get(layer, expert, 4) for expert in EXPERTS]
    bank = {}
    for name, (_, meta) in weights[0].items():
        if any(value[name][1] != meta for value in weights):
            raise RuntimeError(f'Mismatched bank layout for {name}')
        bank[name] = pack([value[name][0] for value in weights]), meta
    mx.eval([array for planes, _ in bank.values() for array in planes.values()])
    route = mx.array((7, 2, 3, 0, 5, 1, 6, 4), dtype=mx.uint32)
    rows = mx.random.normal((8, 4096)).astype(mx.float16).astype(mx.float32)
    mx.eval(route, rows)
    guard.check(force=True)
    results = []
    for mode in ('base', 'refined', 'mixed', 'mixed-cold'):
        level = 4 if mode == 'refined' else 2
        refined = (mx.where(mx.arange(8) % 2 == 0, route, mx.array(0xffffffff, dtype=mx.uint32))
                   if mode == 'mixed' else mx.full((8,), 0xffffffff, dtype=mx.uint32))
        mx.eval(refined)

        def run(options):
            def project(name, value):
                planes, meta = bank[name]
                extra = dict(refinements=planes, refined_route=refined) if mode.startswith('mixed') else {}
                return projection_matvec_banked(planes, meta, value, route, level,
                                                 **options, **extra)

            gate = mx.minimum(project('gate', rows), 10.0)
            up = mx.clip(project('up', rows), -10.0, 10.0)
            return project('down', gate * mx.sigmoid(gate) * up)

        expected = run({})
        mx.eval(expected)
        if not mx.isfinite(expected).all().item():
            raise RuntimeError('Nonfinite banked reference')
        for name, options in variants.items():
            for _ in range(5):
                actual = run(options)
                mx.eval(actual)
                if not mx.array_equal(actual, expected).item():
                    raise RuntimeError(f'Banked output changed: layer {layer}, {mode}, {name}')
            guard.check(force=True)
        samples = {name: [] for name in variants}
        schedule = random.Random(42)
        for _ in range(25):
            order = list(variants)
            schedule.shuffle(order)
            for name in order:
                started = time.perf_counter()
                actual = run(variants[name])
                mx.eval(actual)
                samples[name].append((time.perf_counter() - started) * 1000)
                if not mx.array_equal(actual, expected).item():
                    raise RuntimeError('Timed banked output changed')
            guard.check(force=True)
        results.append(dict(mode=mode, identical=True,
                            variants=[dict(**variants[name], median_ms=statistics.median(values),
                                           samples_ms=values) for name, values in samples.items()]))
    return dict(layer=layer, scenarios=results)


def compare_stages(store, layer, guard):
    from unittest.mock import patch

    import decoder

    weights = [store.get(layer, expert, 4) for expert in EXPERTS]
    route = mx.array((7, 2, 3, 0, 5, 1, 6, 4), dtype=mx.uint32)
    records = []
    for name, (_, meta) in weights[0].items():
        planes = pack([expert[name][0] for expert in weights])
        inputs = mx.random.normal((8, meta['k'])).astype(mx.float16).astype(mx.float32)
        mx.eval(planes, inputs, route)
        for mode in ('base', 'refined', 'mixed', 'mixed-cold'):
            level = 4 if mode == 'refined' else 2
            refined = (mx.where(mx.arange(8) % 2 == 0, route, mx.array(0xffffffff, dtype=mx.uint32))
                       if mode == 'mixed' else mx.full((8,), 0xffffffff, dtype=mx.uint32))
            extra = dict(refinements=planes, refined_route=refined) if mode.startswith('mixed') else {}
            mx.eval(refined)

            def project(value):
                return projection_matvec_banked(planes, meta, value, route, level, **extra)

            captured = []
            factory = decoder._matvec_kernel

            def capture(*args):
                kernel = factory(*args)

                def invoke(**kwargs):
                    result = kernel(**kwargs)
                    captured.append((kernel, kwargs, result))
                    return result

                return invoke

            with patch.object(decoder, '_matvec_kernel', capture):
                expected = project(inputs)
                mx.eval(expected)
            if len(captured) != 1 or not mx.isfinite(expected).all().item():
                raise RuntimeError('Projection did not capture one finite matvec')
            kernel, arguments, partial = captured.pop()
            mx.eval(arguments['inputs'], partial)
            compiled = mx.compile(project)
            calls = dict(full=lambda: project(inputs), raw=lambda: kernel(**arguments)[0],
                         compiled=lambda: compiled(inputs))
            outputs = {kind: call() for kind, call in calls.items()}
            mx.eval(outputs)
            if (not mx.array_equal(outputs['raw'], partial[0]).item()
                    or not mx.array_equal(outputs['full'], expected).item()):
                raise RuntimeError('Raw or full projection output changed')
            compiled_relative = (mx.linalg.norm(outputs['compiled'] - expected) / mx.linalg.norm(expected)).item()
            if not compiled_relative < 0.001:
                raise RuntimeError('Compiled projection exceeds the existing relative-error bound')
            for call in calls.values():
                for _ in range(5):
                    mx.eval(call())
            samples = {kind: [] for kind in calls}
            schedule = random.Random(42)
            for _ in range(25):
                order = list(calls)
                schedule.shuffle(order)
                for kind in order:
                    started = time.perf_counter()
                    output = calls[kind]()
                    mx.eval(output)
                    samples[kind].append((time.perf_counter() - started) * 1000)
                    if not mx.array_equal(output, outputs[kind]).item():
                        raise RuntimeError('Timed projection output changed')
                guard.check(force=True)
            records.append(dict(projection=name, mode=mode, compiled_relative_rms=compiled_relative,
                                compiled_identical=mx.array_equal(outputs['compiled'], expected).item(),
                                variants={kind: dict(median_ms=statistics.median(values), samples_ms=values)
                                          for kind, values in samples.items()}))
    return dict(layer=layer, projections=records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    probes = parser.add_mutually_exclusive_group()
    probes.add_argument('--threadgroups', action='store_true')
    probes.add_argument('--stages', action='store_true')
    probes.add_argument('--decoder-variants', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output file')
    mx.set_cache_limit(256 * 1024**2)
    mx.random.seed(42)
    store = ExpertStore(args.model, cache_gib=0.5)
    if args.threadgroups or args.stages or args.decoder_variants:
        from metrics import MemoryGuard

        guard = MemoryGuard(4)
        store.memory_check = lambda: guard.check(force=True)
    try:
        checks = []
        for layer in (3, 44):
            if args.stages:
                check = compare_stages(store, layer, guard)
                checks.append(check)
                print(json.dumps(check), flush=True)
                continue
            if args.threadgroups or args.decoder_variants:
                variants = ({str(size): dict(threadgroup_size=size) for size in (64, 128, 256, 512)}
                            if args.threadgroups else
                            {f'{load}-{lookup}': dict(state_load=load, sum_lookup=lookup)
                             for load, lookup in (('bytes', False), ('u32', False), ('bytes', True), ('u32', True),
                                                  ('shuffle', False), ('shuffle', True))})
                check = compare_variants(store, layer, guard, variants)
                checks.append(check)
                print(json.dumps(check), flush=True)
                continue
            for level in (2, 4):
                check = compare(store, layer, level)
                checks.append(check)
                print(json.dumps(check), flush=True)
    finally:
        store.close()
    with args.output.open('x') as output:
        json.dump(dict(scope='Eight cached experts per layer, with a bank packed before timing',
                       threadgroup_probe=args.threadgroups, stages_probe=args.stages,
                       decoder_variants_probe=args.decoder_variants,
                       raw_scope='Matvec only, with already evaluated rotated inputs; timings are not additive',
                       checks=checks), output, indent=2)
        output.write('\n')


if __name__ == '__main__':
    main()
