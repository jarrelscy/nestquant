from collections import OrderedDict
import fcntl
import json
import math
import mmap
import os
from pathlib import Path
import time

import mlx.core as mx
import numpy as np

from mapped_weights import read_layouts


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


def _shared_plane(rows, shape, dtype):
    row_bytes = math.prod(shape) * dtype.size
    if row_bytes <= 0:
        raise ValueError('A packed expert plane must contain data')
    mapping = mmap.mmap(-1, row_bytes * rows)
    raw = np.ndarray((rows, row_bytes), dtype=np.uint8, buffer=mapping)
    array = mx.from_dlpack(raw, copy=False).view(dtype).reshape((rows, *shape))
    return mapping, raw, array


class PackedExpertBank:
    def __init__(self, path: Path, layer, rows, revision, remaining_bytes=None):
        if not 32 <= rows <= 288:
            raise ValueError('Expected between 32 and 288 packed rows')
        self.layer = layer
        self.rows = rows
        self.handles = []
        self.slots = OrderedDict()
        self.free_slots = list(reversed(range(rows)))
        self.planes = {}
        self.host = {}
        self.previous_output = None
        self.selection_pending = False
        self.fully_resident = False
        self.prefetch = None
        self.stats = dict(hits=0, misses=0, evictions=0, read_bytes=0, read_seconds=0.0, resident_routes=0,
                          prefetch_hits=0, prefetch_wait_seconds=0.0, prefetch_copy_seconds=0.0,
                          prefetch_read_bytes=0, prefetch_read_seconds=0.0, prefetch_discarded=0,
                          prefetch_cancelled=0, prefetch_errors=0, preload_sleep_seconds=0.0)
        self.fd = path.open('rb')
        try:
            fcntl.fcntl(self.fd, fcntl.F_NOCACHE, 1)
            size = os.fstat(self.fd.fileno()).st_size
            layouts, metadata, offset = read_layouts(self.fd, size)
            if (metadata['format'], metadata['revision'], metadata['layer']) != (
                    'nestquant-resident-base-v1', revision, str(layer)):
                raise ValueError('Unexpected packed expert archive metadata')
            geometry = json.loads(metadata['geometry'])
            self.planes = {name: ({}, meta) for name, meta in geometry.items()}
            if any(not shape or shape[0] != 288 for _, _, _, shape in layouts.values()):
                raise ValueError('Unexpected packed expert geometry')
            self.bytes = rows * sum(math.prod(shape[1:]) * dtype.size
                                    for _, _, dtype, shape in layouts.values())
            if remaining_bytes is not None and self.bytes > remaining_bytes:
                raise MemoryError('Native base banks exceed the packed row budget')
            for name, (start, _, dtype, shape) in layouts.items():
                projection, field = name.split('.', 1)
                row_bytes = math.prod(shape[1:]) * dtype.size
                mapping, raw, array = _shared_plane(rows, shape[1:], dtype)
                self.handles.append(mapping)
                self.host[name] = raw, start + offset, row_bytes
                self.planes[projection][0][field] = array
            mx.eval([array for planes, _ in self.planes.values() for array in planes.values()])
        except BaseException:
            self.close()
            raise

    def _fill(self, slot, expert):
        start = time.perf_counter()
        for target, offset, row_bytes in self.host.values():
            position = offset + expert * row_bytes
            with memoryview(target[slot]) as row:
                written = 0
                while written < row_bytes:
                    count = os.preadv(self.fd.fileno(), [row[written:]], position + written)
                    if count == 0:
                        raise EOFError('Packed expert read ended early')
                    written += count
            self.stats['read_bytes'] += row_bytes
        self.stats['read_seconds'] += time.perf_counter() - start

    def read_payload(self, expert):
        start = time.perf_counter()
        payload = {}
        for name, (_, offset, row_bytes) in self.host.items():
            payload[name] = bytearray(row_bytes)
            with memoryview(payload[name]) as row:
                written = 0
                while written < row_bytes:
                    count = os.preadv(self.fd.fileno(), [row[written:]], offset + expert * row_bytes + written)
                    if count == 0:
                        raise EOFError('Packed expert read-ahead ended early')
                    written += count
        return payload, sum(len(value) for value in payload.values()), time.perf_counter() - start

    def _install_payload(self, slot, payload):
        if set(payload) != set(self.host) or any(len(payload[name]) != size for name, (_, _, size) in self.host.items()):
            raise ValueError('Missing or malformed packed base payload')
        for name, (target, _, _) in self.host.items():
            target[slot] = np.frombuffer(payload[name], dtype=np.uint8)

    def preload(self, check, bytes_per_second=0):
        if self.rows != 288 or self.slots or self.selection_pending:
            raise RuntimeError('Preload requires an unused bank with all 288 expert rows')
        if not math.isfinite(bytes_per_second) or bytes_per_second < 0:
            raise ValueError('Preload rate must be finite and nonnegative')
        start = time.perf_counter()
        loaded = 0
        for target, offset, _ in self.host.values():
            with memoryview(target).cast('B') as buffer:
                written = 0
                while written < buffer.nbytes:
                    end = min(buffer.nbytes, written + 16 * 1024**2)
                    read_start = time.perf_counter()
                    count = os.preadv(self.fd.fileno(), [buffer[written:end]], offset + written)
                    self.stats['read_seconds'] += time.perf_counter() - read_start
                    if count == 0:
                        raise EOFError('Packed expert preload ended early')
                    written += count
                    loaded += count
                    self.stats['read_bytes'] += count
                    check()
                    if bytes_per_second:
                        delay = loaded / bytes_per_second - (time.perf_counter() - start)
                        if delay > 0:
                            wait_start = time.perf_counter()
                            time.sleep(delay)
                            self.stats['preload_sleep_seconds'] += time.perf_counter() - wait_start
                            check()
        checks = []
        for planes, _ in self.planes.values():
            for array in planes.values():
                indices = mx.arange(0, array.size, 16384 // array.itemsize)
                checks.append(array.reshape(-1)[indices].astype(mx.float32).sum())
        mx.eval(checks)
        check()
        self.slots.update((expert, expert) for expert in range(288))
        self.free_slots.clear()
        self.fully_resident = True

    def select_device(self, experts):
        if not self.fully_resident:
            raise RuntimeError('Device routing requires a fully loaded base bank')
        if self.selection_pending:
            raise RuntimeError('Retain the selected rows\' output before selecting again')
        if experts.ndim != 1 or experts.dtype != mx.uint32 or experts.size == 0:
            raise ValueError('Device routes require a nonempty uint32 vector')
        self.selection_pending = True
        self.stats['resident_routes'] += experts.size
        return experts

    def select(self, experts):
        if self.selection_pending:
            raise RuntimeError('Retain the selected rows\' output before selecting again')
        ordered = tuple(dict.fromkeys(experts))
        if len(ordered) > self.rows or any(type(expert) is not int or not 0 <= expert < 288 for expert in ordered):
            raise ValueError('The routed experts do not fit the packed row bank')
        if any(expert not in self.slots for expert in ordered):
            self.finish()
        wanted = set(ordered)
        for expert in ordered:
            if expert in self.slots:
                self.slots.move_to_end(expert)
                self.stats['hits'] += 1
                continue
            if self.free_slots:
                slot = self.free_slots.pop()
            else:
                evict = next(key for key in self.slots if key not in wanted)
                slot = self.slots.pop(evict)
                self.stats['evictions'] += 1
            try:
                if self.prefetch is None or not self.prefetch.consume(self.layer, expert, slot):
                    self._fill(slot, expert)
            except BaseException:
                self.free_slots.append(slot)
                raise
            self.slots[expert] = slot
            self.stats['misses'] += 1
        if self.prefetch is not None:
            self.prefetch.advance(self.layer)
        self.selection_pending = True
        return tuple(self.slots[expert] for expert in experts)

    def retain(self, output):
        if not self.selection_pending:
            raise RuntimeError('Select expert rows before retaining an output')
        self.previous_output = output
        self.selection_pending = False

    def finish(self):
        if self.previous_output is not None:
            mx.eval(self.previous_output)
            self.previous_output = None

    def close(self):
        try:
            self.finish()
            mx.synchronize()
        finally:
            self.fully_resident = False
            self.fd.close()
            self.planes.clear()
            self.host.clear()
            self.handles.clear()


class PackedRefinementBank:
    @staticmethod
    def fields(base):
        for name, (base_planes, meta) in base.items():
            units = meta['tk'] * meta['tn']
            fields = {
                'p4': ((units * int(meta['res_rule']['K'] * 256),), mx.uint8, np.dtype('u1')),
                'word': ((units,), mx.int32, np.dtype('<i4')),
                'suh4': ((meta['k'],), mx.float16, np.dtype('<f2')),
                'svh4': ((meta['n'],), mx.float16, np.dtype('<f2')),
            }
            if 'lrV' in base_planes:
                fields['lrU4'] = ((base_planes['lrV'].shape[1], meta['n']), mx.float16, np.dtype('<f2'))
            yield name, fields

    @classmethod
    def bytes_for(cls, base, rows):
        return rows * sum(math.prod(shape) * dtype.size for _, fields in cls.fields(base)
                          for shape, dtype, _ in fields.values())

    def __init__(self, base, rows):
        if not 1 <= rows <= 240:
            raise ValueError('Expected between 1 and 240 refinement rows')
        self.slots = {}
        self.available = list(reversed(range(rows)))
        self.host = {}
        self.planes = {}
        self.handles = []
        self.previous_output = None
        self.bytes = self.bytes_for(base, rows)
        for name, fields in self.fields(base):
            self.planes[name] = {}
            self.host[name] = {}
            for field, (shape, dtype, np_dtype) in fields.items():
                mapping, raw, array = _shared_plane(rows, shape, dtype)
                self.planes[name][field] = array
                self.host[name][field] = raw.view(np_dtype).reshape((rows, *shape))
                self.handles.append(mapping)
        mx.eval([array for planes in self.planes.values() for array in planes.values()])

    def install(self, expert, payload, wanted):
        if expert in self.slots:
            return True
        victim = None
        if self.available:
            slot = self.available[-1]
        else:
            victim = next((key for key in self.slots if key not in wanted), None)
            if victim is None:
                return False
            slot = self.slots[victim]
        self.finish()
        if victim is None:
            self.available.pop()
        else:
            del self.slots[victim]
        try:
            for name, fields in self.host.items():
                if set(payload[name]) - set(fields):
                    raise ValueError('Unexpected refinement plane')
                for field, storage in fields.items():
                    value = payload[name].get(field)
                    target = storage[slot]
                    if field == 'lrU4':
                        target.fill(0)
                        if value is not None:
                            if value.ndim != 2 or value.shape[0] > target.shape[0] or value.shape[1] != target.shape[1]:
                                raise ValueError('Unexpected low-rank refinement shape')
                            np.copyto(target[:value.shape[0]], value, casting='no')
                    elif value is None or value.shape != target.shape:
                        raise ValueError('Missing or malformed refinement plane')
                    else:
                        np.copyto(target, value, casting='no')
        except BaseException:
            self.available.append(slot)
            raise
        self.slots[expert] = slot
        return True

    def retain(self, output):
        self.previous_output = output

    def finish(self):
        if self.previous_output is not None:
            mx.eval(self.previous_output)
            self.previous_output = None
