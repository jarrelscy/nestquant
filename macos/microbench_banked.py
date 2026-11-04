import argparse
import json
from pathlib import Path
import statistics
import time

import mlx.core as mx

from decoder import projection_matvec, projection_matvec_banked
from experts import ExpertStore


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


def pack(projections):
    low_rank = {'lrV', 'lrU2', 'lrU4'}
    packed = {}
    for key in sorted(set().union(*(planes.keys() for planes in projections))):
        if key not in low_rank:
            packed[key] = mx.stack([planes[key] for planes in projections])
            continue
        present = [planes[key] for planes in projections if key in planes]
        rank = max(array.shape[0] for array in present)
        _, width = present[0].shape
        dtype = present[0].dtype
        padded = []
        for planes in projections:
            value = planes.get(key)
            if value is None:
                value = mx.zeros((rank, width), dtype=dtype)
            else:
                value = mx.pad(value, [(0, rank - value.shape[0]), (0, 0)])
            padded.append(value)
        packed[key] = mx.stack(padded)
    return packed


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

    def batched(indices):
        gate = projection_matvec_banked(*bank['gate'], rows, indices, level)
        up = projection_matvec_banked(*bank['up'], rows, indices, level)
        return projection_matvec_banked(*bank['down'], activation(gate, up), indices, level)

    expected = mx.stack([store.apply(layer, EXPERTS[index], rows[row], level) for row, index in enumerate(route)])
    device_route = mx.array(route, dtype=mx.uint32)
    actual = batched(route)
    on_device = batched(device_route)
    mx.eval(expected, actual, on_device)
    relative_rms = (mx.linalg.norm(actual - expected) / mx.linalg.norm(expected)).item()
    if not relative_rms < 0.001:
        raise RuntimeError(f'Banked expert parity failed at layer {layer}, level {level}: {relative_rms}')
    if not mx.array_equal(actual, on_device).item():
        raise RuntimeError('Device routes changed the banked result')
    invalid = batched(mx.array((*route[:-1], len(EXPERTS)), dtype=mx.uint32))
    if not mx.isfinite(invalid[:-1]).all().item() or not mx.isnan(invalid[-1]).all().item():
        raise RuntimeError('Invalid device route did not fail in its own row')
    return dict(layer=layer, level=level, relative_rms=relative_rms,
                serial=timed(lambda: (serial() * scores[:, None]).sum(axis=0)),
                banked=timed(lambda: (batched(route) * scores[:, None]).sum(axis=0)),
                device=timed(lambda: (batched(device_route) * scores[:, None]).sum(axis=0)))


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
