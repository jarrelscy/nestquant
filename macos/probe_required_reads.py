import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import time

from expert_major_base import plane_layouts, read_into
from mapped_weights import read_layouts
from metrics import MemoryGuard
from native_assets import REVISION, fingerprint, paths
from resources import process_usage, system_pressure, vm_snapshot


def read_row(row, storage):
    with memoryview(storage)[:row['bytes']] as target:
        started = time.perf_counter()
        count = read_into(row['fd'], [target], row['offset'])
        seconds = time.perf_counter() - started
        if (count != row['bytes'] or bytes(target[:64]) != row['first']
                or bytes(target[-64:]) != row['last']):
            raise RuntimeError('Required read returned the wrong expert row')
    return count, seconds


def read_batches(executor, rows, width, buffers, check):
    total_bytes, active_seconds, latencies = 0, 0.0, []
    for start in range(0, len(rows), width):
        check()
        batch = rows[start:start + width]
        started = time.perf_counter()
        futures = [executor.submit(read_row, row, buffers[index]) for index, row in enumerate(batch)]
        results = [future.result() for future in futures]
        active_seconds += time.perf_counter() - started
        for count, seconds in results:
            total_bytes += count
            latencies.append(seconds)
        check()
    return dict(bytes=total_bytes, active_batch_seconds=active_seconds, read_seconds=latencies)


def probe(archive, reads, check):
    coordinates = random.Random(0).sample([(layer, expert) for layer in range(3, 45) for expert in range(288)],
                                        reads + 8)
    trace_sha256 = hashlib.sha256(json.dumps(coordinates).encode()).hexdigest()
    files, rows, outcomes = {}, [], []
    with ExitStack() as stack:
        for layer in range(3, 45):
            file, sidecar = paths(archive, layer)
            evidence = json.loads(sidecar.read_text())
            if evidence['revision'] != REVISION or evidence['state'] != fingerprint(file):
                raise RuntimeError('Expert archive fingerprint changed')
            reader = stack.enter_context(file.open('rb', buffering=0))
            fcntl.fcntl(reader, fcntl.F_NOCACHE, 1)
            layouts, metadata, data_start = read_layouts(reader, os.fstat(reader.fileno()).st_size)
            _, size = plane_layouts(layouts, metadata)
            if metadata.get('revision') != REVISION or metadata.get('layer') != str(layer) or size < 128:
                raise ValueError('Unexpected expert archive metadata')
            files[layer] = reader, data_start, size, file, evidence['state']
        for layer, expert in coordinates:
            reader, start, size, _, _ = files[layer]
            offset = start + expert * size
            rows.append(dict(fd=reader.fileno(), offset=offset, bytes=size,
                             first=os.pread(reader.fileno(), 64, offset),
                             last=os.pread(reader.fileno(), 64, offset + size - 64)))
        buffers = [bytearray(max(item['bytes'] for item in rows)) for _ in range(4)]
        check()
        for sequence, width in enumerate((1, 2, 4, 4, 2, 1)):
            with ThreadPoolExecutor(max_workers=width, thread_name_prefix='required-read-probe') as executor:
                read_batches(executor, rows[:8], width, buffers, check)
                before = process_usage().diskio_bytesread
                started = time.perf_counter()
                result = read_batches(executor, rows[8:], width, buffers, check)
                result['wall_seconds'] = time.perf_counter() - started
                result['process_disk_read_bytes'] = process_usage().diskio_bytesread - before
            result.update(sequence=sequence, width=width, reads=reads,
                          active_batch_GB_s=result['bytes'] / result['active_batch_seconds'] / 1e9,
                          wall_GB_s=result['bytes'] / result['wall_seconds'] / 1e9,
                          median_read_seconds=statistics.median(result['read_seconds']))
            outcomes.append(result)
            print(json.dumps({key: result[key] for key in ('sequence', 'width', 'wall_GB_s',
                                                           'active_batch_GB_s', 'median_read_seconds')}), flush=True)
        for _, _, _, file, state in files.values():
            if fingerprint(file) != state:
                raise RuntimeError('Expert archive changed during the read probe')
    return dict(trace_sha256=trace_sha256, warmup_reads_per_trial=8, measurements=outcomes,
                buffer_bytes=sum(len(buffer) for buffer in buffers),
                limitations=['This measures CPU reads into private buffers, not full-model throughput or GPU-bank writes.',
                             'Every trial uses the same distinct rows and uncached handles; operating-system caching is not assumed absent.',
                             'Boundary samples verify row addressing; the separate archive verifier supplies complete byte equality.',
                             'Active batch time includes dispatch, completion, and boundary validation. Wall time also includes guard checks.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reads', type=int, choices=(128, 384, 768), default=384)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output path')
    guard = MemoryGuard(4)
    before = vm_snapshot()
    result = probe(args.archive, args.reads, lambda: guard.check(force=True))
    usage = process_usage()
    result.update(status='passed', memory=dict(physical_bytes=usage.phys_footprint,
                                              lifetime_peak_physical_bytes=usage.lifetime_max_phys_footprint,
                                              pressure_level=system_pressure()),
                  vm_before=before, vm_after=vm_snapshot())
    with args.output.open('x') as output:
        json.dump(result, output, indent=2)
        output.write('\n')
