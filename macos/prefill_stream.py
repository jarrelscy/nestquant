"""Stream 4-bit refinement planes for batched prefill straight from the checkpoint shards into a staging ring.

RefinementReader.read assembles each expert with many small reads, concatenations and copies under the
GIL. Here each expert's tensors are read with os.preadv directly into one row of a shared, MLX-visible
staging buffer, so reader threads spend their time in the kernel and the GPU reads the row in place.
Replicated tensors (the unsplit scale and the down projection's low-rank plane) are read from shard 0
only. The layout of a row matches PackedRefinementBank: p4, word, suh4, svh4 and lrU4 per projection,
with lrU4 zero padded to MAX_RANK rows.

RefinementStream reads a fixed sequence of (layer, expert) keys ahead of the consumer, bounded by the
number of staging rows. The consumer takes keys in sequence order, gives rows back once the GPU has
copied what it needs, and can skip keys it turns out not to need.
"""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
import math
import mmap
import os
import threading
import time

import mlx.core as mx
import numpy as np

from positional_reader import PositionalTensorReader

MAX_RANK = 4
SHARDS = 8
PROJECTIONS = ('gate', 'up', 'down')
LAYOUT_KEYS = ('k', 'n', 'tk', 'tn', 'shard_axis', 'base_K', 'res_rule')


def _shared(rows, shape, dtype, np_dtype):
    row_bytes = math.prod(shape) * dtype.size
    mapping = mmap.mmap(-1, row_bytes * rows)
    raw = np.ndarray((rows, row_bytes), dtype=np.uint8, buffer=mapping)
    array = mx.from_dlpack(raw, copy=False).view(dtype).reshape((rows, *shape))
    return mapping, raw.view(np_dtype).reshape((rows, *shape)), array


class RefinementStaging:
    """Fixed rows of refinement planes in shared memory, visible to MLX without copies."""

    def __init__(self, geometry, rows):
        self.rows = rows
        self.geometry = geometry
        self.planes = {}
        self.host = {}
        self.handles = []
        self.bytes = 0
        for name, meta in geometry.items():
            units = meta['tk'] * meta['tn']
            fields = {
                'p4': ((units * int(meta['res_rule']['K'] * 256),), mx.uint8, np.dtype('u1')),
                'word': ((units,), mx.int32, np.dtype('<i4')),
                'suh4': ((meta['k'],), mx.float16, np.dtype('<f2')),
                'svh4': ((meta['n'],), mx.float16, np.dtype('<f2')),
                'lrU4': ((MAX_RANK, meta['n']), mx.float16, np.dtype('<f2')),
            }
            self.planes[name] = {}
            self.host[name] = {}
            for field, (shape, dtype, np_dtype) in fields.items():
                mapping, host, array = _shared(rows, shape, dtype, np_dtype)
                self.handles.append(mapping)
                self.planes[name][field] = array
                self.host[name][field] = host
                self.bytes += host.nbytes
        mx.eval([array for planes in self.planes.values() for array in planes.values()])

    @staticmethod
    def bytes_per_row(geometry):
        total = 0
        for meta in geometry.values():
            units = meta['tk'] * meta['tn']
            total += units * int(meta['res_rule']['K'] * 256) + 4 * units + 2 * (meta['k'] + meta['n'])
            total += 2 * MAX_RANK * meta['n']
        return total

    def close(self):
        self.planes.clear()
        self.host.clear()
        for mapping in self.handles:
            try:
                mapping.close()
            except BufferError:
                pass
        self.handles.clear()


class ShardedRefinementReader:
    """Read one expert's refinement planes from the eight tensor-parallel shards into a staging row."""

    def __init__(self, root):
        self.root = root
        self.lock = threading.Lock()
        self.handles = {}
        self.manifests = {}
        self.geometry = None

    def manifest(self, layer):
        with self.lock:
            if layer not in self.manifests:
                path = self.root / 'layers' / f'L{layer}' / 'manifest.json'
                manifest = json.loads(path.read_text())
                geometry = {name: {key: manifest['proj_meta'][name][key] for key in LAYOUT_KEYS}
                            for name in PROJECTIONS}
                if self.geometry is None:
                    self.geometry = geometry
                elif geometry != self.geometry:
                    raise ValueError('Refinement geometry differs between layers')
                self.manifests[layer] = manifest
            return self.manifests[layer]

    def _handle(self, layer, shard):
        with self.lock:
            key = layer, shard
            if key not in self.handles:
                self.handles[key] = PositionalTensorReader(
                    self.root / 'layers' / f'L{layer}' / f'tp{shard}.safetensors')
            return self.handles[key]

    def bytes_for(self, layer, expert):
        manifest = self.manifest(layer)
        ranks = manifest['per_expert'][str(expert)]['lr_rank']
        total = 0
        for name, meta in manifest['proj_meta'].items():
            units = meta['tk'] * meta['tn']
            total += units * (int(meta['res_rule']['K'] * 256) + 4) + 2 * (meta['k'] + meta['n'])
            total += 2 * ranks[name] * meta['n']
        return total

    def read_into(self, layer, expert, staging, slot):
        """Fill staging row `slot`. Returns the number of bytes read."""
        started = time.perf_counter()
        manifest = self.manifest(layer)
        ranks = manifest['per_expert'][str(expert)]['lr_rank']
        total = 0
        temporaries = []
        for shard in range(SHARDS):
            handle = self._handle(layer, shard)
            pieces = []
            for name in PROJECTIONS:
                meta = manifest['proj_meta'][name]
                host = staging.host[name]
                rank = ranks[name]
                if rank > MAX_RANK:
                    raise ValueError('Refinement low-rank plane exceeds the staging rank')
                for field in ('p4', 'word'):
                    row = host[field][slot]
                    part = row.size // SHARDS
                    pieces.append((f'E{expert}.{name}.{field}', row[shard * part:(shard + 1) * part]))
                for axis in ('su', 'sv'):
                    field = f'{axis}h4'
                    row = host[field][slot]
                    split = (name == 'down') == (axis == 'su')
                    if split:
                        part = row.size // SHARDS
                        pieces.append((f'E{expert}.{name}.{field}', row[shard * part:(shard + 1) * part]))
                    elif shard == 0:
                        pieces.append((f'E{expert}.{name}.{field}', row))
                if rank:
                    target = host['lrU4'][slot]
                    if name == 'down':
                        if shard == 0:
                            pieces.append((f'E{expert}.{name}.lrU4', target[:rank]))
                    else:
                        width = meta['n'] // SHARDS
                        temporary = np.empty((rank, width), dtype=np.float16)
                        temporaries.append((temporary, target[:rank, shard * width:(shard + 1) * width]))
                        pieces.append((f'E{expert}.{name}.lrU4', temporary))
            total += self._read_pieces(handle, pieces)
        for temporary, target in temporaries:
            np.copyto(target, temporary, casting='no')
        for name in PROJECTIONS:
            rank = ranks[name]
            staging.host[name]['lrU4'][slot][rank:] = 0
            if np.any((staging.host[name]['word'][slot] & 255) == 0):
                raise ValueError('Invalid zero denominator in refinement block')
        return total, time.perf_counter() - started

    @staticmethod
    def _read_pieces(handle, pieces):
        ranges = []
        for key, target in pieces:
            begin, end, _, dtype, shape = handle._tensor_range(key)
            view = memoryview(target).cast('B') if target.flags.c_contiguous else None
            if view is None or view.nbytes != end - begin:
                raise ValueError(f'Refinement tensor {key} does not match its staging row')
            ranges.append((begin, end, view))
        ranges.sort(key=lambda item: item[0])
        total = 0
        position = 0
        while position < len(ranges):
            begin, end, view = ranges[position]
            buffers = [view]
            position += 1
            # Coalesce neighbouring tensors into one vectored read when the gap between them is small.
            while position < len(ranges) and ranges[position][0] - end <= 16384:
                gap = ranges[position][0] - end
                if gap:
                    buffers.append(bytearray(gap))
                buffers.append(ranges[position][2])
                end = ranges[position][1]
                position += 1
            offset = handle.data_start + begin
            remaining = end - begin
            while remaining:
                count = os.preadv(handle.fd, buffers, offset)
                if count <= 0:
                    raise EOFError('Refinement read ended early')
                total += count
                remaining -= count
                offset += count
                if remaining:
                    # Drop the buffers that were filled completely and trim the partial one.
                    filled = count
                    while buffers and filled >= len(buffers[0]):
                        filled -= len(buffers[0])
                        buffers.pop(0)
                    if filled:
                        buffers[0] = memoryview(buffers[0])[filled:]
        return total

    def close(self):
        with self.lock:
            for handle in self.handles.values():
                handle.close()
            self.handles.clear()


class RefinementStream:
    """Read a sequence of (layer, expert) refinements ahead of the consumer into a bounded staging ring."""

    def __init__(self, reader, staging, keys, threads=8):
        self.reader = reader
        self.staging = staging
        self.queue = deque(keys)
        self.skipped = set()
        self.free = list(reversed(range(staging.rows)))
        self.inflight = {}
        self.pool = ThreadPoolExecutor(max_workers=threads, thread_name_prefix='nestquant-prefill-stream')
        self.stats = dict(read_bytes=0, reads=0, read_seconds=0.0, wait_seconds=0.0, skipped=0)
        self._top_up()

    def _top_up(self):
        while self.free and self.queue:
            key = self.queue.popleft()
            if key in self.skipped:
                self.skipped.discard(key)
                self.stats['skipped'] += 1
                continue
            slot = self.free.pop()
            self.inflight[key] = slot, self.pool.submit(self.reader.read_into, *key, self.staging, slot)

    def skip(self, keys):
        """Drop keys the consumer will not take. Rows already read for them are released."""
        for key in keys:
            if key in self.inflight:
                slot, future = self.inflight.pop(key)
                future.result()
                self.free.append(slot)
                self.stats['skipped'] += 1
            else:
                self.skipped.add(key)
        self._top_up()

    def take(self, key):
        """Wait for one key's row and return its staging slot. Keys must be taken in sequence order."""
        if key not in self.inflight:
            self._top_up()
        if key not in self.inflight:
            raise KeyError(f'Refinement {key} is not queued')
        slot, future = self.inflight.pop(key)
        started = time.perf_counter()
        count, seconds = future.result()
        self.stats['wait_seconds'] += time.perf_counter() - started
        self.stats['read_bytes'] += count
        self.stats['read_seconds'] += seconds
        self.stats['reads'] += 1
        return slot

    def release(self, slots):
        self.free.extend(slots)
        self._top_up()

    def close(self):
        self.queue.clear()
        for slot, future in self.inflight.values():
            try:
                future.result()
            except BaseException:
                pass
        self.inflight.clear()
        self.pool.shutdown(wait=True)
