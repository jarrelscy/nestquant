import argparse
import json
from pathlib import Path
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output file')
    mx.set_cache_limit(256 * 1024**2)
    mx.random.seed(42)
    store = ExpertStore(args.model, cache_gib=0.5)
    try:
        checks = []
        for layer in (3, 44):
            for level in (2, 4):
                check = compare(store, layer, level)
                checks.append(check)
                print(json.dumps(check), flush=True)
    finally:
        store.close()
    with args.output.open('x') as output:
        json.dump(dict(scope='Eight cached experts per layer, with a bank packed before timing', checks=checks), output, indent=2)
        output.write('\n')


if __name__ == '__main__':
    main()
