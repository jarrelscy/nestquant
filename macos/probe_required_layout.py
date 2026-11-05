import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import time

import mlx.core as mx

from expert_major_base import read_into
from metrics import MemoryGuard
from native_assets import REVISION, fingerprint, paths
from packed_expert_bank import PackedExpertBank, _shared_plane
from resources import process_usage, system_pressure, vm_snapshot


VARIANTS = ('scatter', 'private', 'shared', 'staged')


def make_trace(layers):
    generator = random.Random(17)
    return {layer: generator.sample(range(288), 32) for layer in layers}


def measure_layer(bank, experts, order, check):
    size = bank.expert_row_bytes
    if (size is None or len(experts) != 32 or len(set(experts)) != 32
            or any(type(expert) is not int or not 0 <= expert < 288 for expert in experts)
            or len(order) != 4 or set(order) != set(VARIANTS)):
        raise ValueError('Expected four variants and 32 distinct expert-major rows')
    private = [bytearray(size) for _ in range(8)]
    staging = bytearray(size)
    mapping, raw, shared = _shared_plane(8, (size,), mx.uint8)
    payload = {}
    results = []
    try:
        mx.eval(shared)
        mx.synchronize()
        for target, _, _ in bank.host.values():
            target[:16].fill(0)
        raw.fill(0)
        for storage in [*private, staging]:
            for position in range(0, size, 4096):
                storage[position] = 0
        for name, (_, offset, length) in bank.host.items():
            start = offset - bank.data_offset
            payload[name] = memoryview(staging)[start:start + length]
        check()
        for variant in order:
            check()
            block = experts[VARIANTS.index(variant) * 8:VARIANTS.index(variant) * 8 + 8]
            reads, copies = [], []
            disk_before = process_usage().diskio_bytesread
            started = time.perf_counter()
            for index, expert in enumerate(block):
                offset = bank.data_offset + expert * size
                tick = time.perf_counter()
                if variant == 'scatter':
                    bank._fill(index, expert)
                elif variant == 'shared':
                    read_into(bank.fd.fileno(), [raw[index]], offset)
                else:
                    read_into(bank.fd.fileno(), [private[index] if variant == 'private' else staging], offset)
                reads.append(time.perf_counter() - tick)
                if variant == 'staged':
                    tick = time.perf_counter()
                    bank._install_payload(8 + index, payload)
                    copies.append(time.perf_counter() - tick)
            wall = time.perf_counter() - started
            disk = process_usage().diskio_bytesread - disk_before
            check()
            results.append(dict(layer=bank.layer, variant=variant, expert_ids=block,
                                bytes=8 * size, process_disk_read_bytes=disk,
                                physical_requested_ratio=disk / (8 * size),
                                disk_admissible=disk >= .95 * 8 * size,
                                read_seconds=reads, copy_seconds=copies, wall_seconds=wall))
        reference = bytearray(size)
        checked = 0
        for variant in VARIANTS:
            block = experts[VARIANTS.index(variant) * 8:VARIANTS.index(variant) * 8 + 8]
            for index in (0, 7):
                check()
                read_into(bank.fd.fileno(), [reference], bank.data_offset + block[index] * size)
                with memoryview(reference) as expected:
                    if variant in ('scatter', 'staged'):
                        slot = index if variant == 'scatter' else index + 8
                        for target, offset, length in bank.host.values():
                            start = offset - bank.data_offset
                            with memoryview(target[slot]).cast('B') as actual:
                                if actual != expected[start:start + length]:
                                    raise RuntimeError('Packed destination differs from the source row')
                    else:
                        with memoryview(private[index] if variant == 'private' else raw[index]).cast('B') as actual:
                            if actual != expected:
                                raise RuntimeError('Contiguous destination differs from the source row')
                checked += 1
        return dict(measurements=results, validated_rows=checked,
                    logical_buffer_bytes=bank.bytes + 18 * size)
    finally:
        for view in payload.values():
            view.release()
        payload.clear()
        mx.synchronize()
        del raw, shared
        mapping.close()


def probe(archive, check):
    trace = make_trace(range(3, 45))
    outcomes, validated, peak_buffers = [], 0, 0
    for sequence, (layer, experts) in enumerate(trace.items()):
        file, sidecar = paths(archive, layer)
        evidence = json.loads(sidecar.read_text())
        if evidence['revision'] != REVISION or evidence['state'] != fingerprint(file):
            raise RuntimeError('Expert archive fingerprint changed')
        bank = PackedExpertBank(file, layer, 32, REVISION, 512 * 1024**2)
        try:
            rotation = sequence % len(VARIANTS)
            order = VARIANTS[rotation:] + VARIANTS[:rotation]
            result = measure_layer(bank, experts, order, check)
            if fingerprint(file) != evidence['state']:
                raise RuntimeError('Expert archive changed during the layout probe')
            outcomes.extend(result['measurements'])
            validated += result['validated_rows']
            peak_buffers = max(peak_buffers, result['logical_buffer_bytes'])
            print(json.dumps(dict(layer=layer, validated_rows=result['validated_rows'],
                                  admissible_variants=[m['variant'] for m in result['measurements']
                                                       if m['disk_admissible']])), flush=True)
        finally:
            bank.close()
        check()
    common = [layer for layer in trace if all(m['disk_admissible'] for m in outcomes if m['layer'] == layer)]
    summaries = {}
    for variant in VARIANTS:
        selected = [m for m in outcomes if m['variant'] == variant and m['layer'] in common]
        reads = [value for m in selected for value in m['read_seconds']]
        copies = [value for m in selected for value in m['copy_seconds']]
        summaries[variant] = dict(comparison_layers=len(selected), total_layers=len(trace),
                                  median_read_seconds=statistics.median(reads) if reads else None,
                                  median_copy_seconds=statistics.median(copies) if copies else None)
    return dict(trace_sha256=hashlib.sha256(json.dumps(trace, sort_keys=True).encode()).hexdigest(),
                measurements=outcomes, summaries=summaries, comparison_layers=common, validated_rows=validated,
                logical_peak_buffer_bytes=peak_buffers,
                limitations=['Single-layer component, without model execution or refinement contention.',
                             'Disjoint timed rows; sampled full-row validation runs after all four variants per layer.',
                             'Summaries include only layers where every variant physically reads at least 95 percent of requested bytes.',
                             'Buffers are touched before timing; model bank slots have a wider range of offsets.',
                             'Component latency does not establish model throughput or answer quality.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output path')
    mx.set_cache_limit(64 * 1024**2)
    guard = MemoryGuard(4)
    before = vm_snapshot()
    result = probe(args.archive, lambda: guard.check(force=True))
    usage = process_usage()
    result.update(status='passed', memory=dict(physical_bytes=usage.phys_footprint,
                                              lifetime_peak_physical_bytes=usage.lifetime_max_phys_footprint,
                                              pressure_level=system_pressure()),
                  vm_before=before, vm_after=vm_snapshot())
    with args.output.open('x') as output:
        json.dump(result, output, indent=2)
        output.write('\n')
