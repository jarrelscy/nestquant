from collections import OrderedDict
import fcntl
import json
import math
import mmap
import os
from pathlib import Path
import threading
import time

import mlx.core as mx
import numpy as np

from mapped_weights import read_layouts
from expert_major_base import FORMAT as EXPERT_MAJOR_FORMAT, MAX_ROW_BYTES, plane_layouts, read_into
from hot_policy import choose_victim


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


MLX_OWNED_RESIDENT_BASE = True


class BaseReadStaging:
    def __init__(self):
        self.storage = bytearray(MAX_ROW_BYTES)
        self.lock = threading.Lock()


class PackedExpertBank:
    def __init__(self, path: Path, layer, rows, revision, remaining_bytes=None, staging=None):
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
        self.staging = staging
        self.expert_row_bytes = None
        self.stats = dict(hits=0, misses=0, evictions=0, read_bytes=0, read_seconds=0.0, resident_routes=0,
                          prefetch_hits=0, prefetch_wait_seconds=0.0, prefetch_copy_seconds=0.0,
                          prefetch_read_bytes=0, prefetch_read_seconds=0.0, prefetch_discarded=0,
                          prefetch_cancelled=0, prefetch_errors=0, preload_sleep_seconds=0.0,
                          read_copy_seconds=0.0, read_staging_wait_seconds=0.0)
        for bucket in (*range(9), 'ge9'):
            self.stats[f'miss_batch_{bucket}_calls'] = 0
            self.stats[f'miss_batch_{bucket}_read_seconds'] = 0.0
        self.fd = path.open('rb')
        try:
            fcntl.fcntl(self.fd, fcntl.F_NOCACHE, 1)
            size = os.fstat(self.fd.fileno()).st_size
            layouts, metadata, offset = read_layouts(self.fd, size)
            if (metadata['format'] not in ('nestquant-resident-base-v1', EXPERT_MAJOR_FORMAT)
                    or (metadata['revision'], metadata['layer']) != (revision, str(layer))):
                raise ValueError('Unexpected packed expert archive metadata')
            self.data_offset = offset
            if metadata['format'] == EXPERT_MAJOR_FORMAT:
                layouts, self.expert_row_bytes = plane_layouts(layouts, metadata)
            if staging is not None and (self.expert_row_bytes is None
                                        or len(staging.storage) < self.expert_row_bytes):
                raise ValueError('Staged reads require an expert-major archive and a complete row buffer')
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
        if self.staging is not None:
            with self.staging.lock:
                self.stats['read_staging_wait_seconds'] += time.perf_counter() - start
                with memoryview(self.staging.storage)[:self.expert_row_bytes] as storage:
                    start = time.perf_counter()
                    count = read_into(self.fd.fileno(), [storage],
                                      self.data_offset + expert * self.expert_row_bytes)
                    self.stats['read_seconds'] += time.perf_counter() - start
                    self.stats['read_bytes'] += count
                    start = time.perf_counter()
                    payload = {name: storage[offset - self.data_offset:offset - self.data_offset + size]
                               for name, (_, offset, size) in self.host.items()}
                    try:
                        self._install_payload(slot, payload)
                    finally:
                        for view in payload.values():
                            view.release()
                    self.stats['read_copy_seconds'] += time.perf_counter() - start
            return
        if self.expert_row_bytes is not None:
            count = read_into(self.fd.fileno(), [target[slot] for target, _, _ in self.host.values()],
                              self.data_offset + expert * self.expert_row_bytes)
            self.stats['read_bytes'] += count
            self.stats['read_seconds'] += time.perf_counter() - start
            return
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
        if self.expert_row_bytes is not None:
            storage = bytearray(self.expert_row_bytes)
            read_into(self.fd.fileno(), [storage], self.data_offset + expert * self.expert_row_bytes)
            payload = {name: memoryview(storage)[offset - self.data_offset:offset - self.data_offset + size]
                       for name, (_, offset, size) in self.host.items()}
            return payload, self.expert_row_bytes, time.perf_counter() - start
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
        if self.expert_row_bytes is not None:
            raise ValueError('Expert-major archives currently support paged execution only')
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
        if MLX_OWNED_RESIDENT_BASE:
            self._adopt_into_mlx(check)
        self.slots.update((expert, expert) for expert in range(288))
        self.free_slots.clear()
        self.fully_resident = True

    def _adopt_into_mlx(self, check):
        # Imported mmap buffers sit outside MLX's wired residency set, so macOS may compress idle base
        # rows. A complete base is never rewritten, so copy each plane into an MLX-owned buffer that
        # mx.set_wired_limit keeps resident, then release the mapping.
        for name, (raw, _, _) in list(self.host.items()):
            projection, field = name.split('.', 1)
            current = self.planes[projection][0][field]
            owned = mx.array(raw).view(current.dtype).reshape(current.shape)
            mx.eval(owned)
            self.planes[projection][0][field] = owned
            del current
            check()
        self.host.clear()
        for mapping in self.handles:
            try:
                mapping.close()
            except BufferError:
                pass
        self.handles.clear()

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
        missing = sum(expert not in self.slots for expert in ordered)
        bucket = missing if missing < 9 else 'ge9'
        reads_before = self.stats['read_seconds']
        if missing:
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
        self.stats[f'miss_batch_{bucket}_calls'] += 1
        self.stats[f'miss_batch_{bucket}_read_seconds'] += self.stats['read_seconds'] - reads_before
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
            self.staging = None
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

    COLD_SLOT = 0xffffffff

    def __init__(self, base, rows):
        if not 1 <= rows <= 240:
            raise ValueError('Expected between 1 and 240 refinement rows')
        self.slots = {}
        self.table = np.full(288, self.COLD_SLOT, dtype=np.uint32)
        self._device_table = None
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

    @property
    def staging(self):
        # expert -> reserved slot whose payload is being written (not yet in the table)
        return self.__dict__.setdefault('_staging', {})

    def install(self, expert, payload, wanted, scores=None):
        if expert in self.slots or expert in self.staging:
            return True
        slot = self.reserve(expert, wanted, scores)
        if slot is None:
            return False
        try:
            self.write(slot, payload)
        except BaseException:
            self.release(expert, slot)
            raise
        self.activate(expert, slot)
        return True

    def reserve(self, expert, wanted, scores=None):
        """Take a slot for `expert`: a free one, else the slot of a resident expert outside `wanted`, which
        leaves the slot table at once. Returns None if the expert is resident or staged, or no slot is free.
        The victim is the oldest install outside `wanted`, or with `scores` ([288] loading priorities) the
        lowest-scoring one (hot_policy.choose_victim).

        Call between steps (no queued GPU work may read the table or the victim's slot). The slot is then
        referenced by no table, so write() may fill it from any thread while later steps run, and activate()
        publishes it at a later step boundary."""
        if expert in self.slots or expert in self.staging:
            return None
        if self.available:
            slot = self.available.pop()
        else:
            victim = choose_victim(self.slots, wanted, scores)
            if victim is None:
                return None
            self.finish()
            slot = self.slots.pop(victim)
            self.table[victim] = self.COLD_SLOT
            self._device_table = None
        self.staging[expert] = slot
        return slot

    def write(self, slot, payload):
        """Copy one refinement payload into a reserved slot (host memory only; safe off the main thread)."""
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

    def activate(self, expert, slot):
        """Publish a written slot. Call between steps, like reserve()."""
        if self.staging.pop(expert) != slot:
            raise RuntimeError('Refinement slot was not reserved for this expert')
        self.slots[expert] = slot
        self.table[expert] = slot
        self._device_table = None

    def release(self, expert, slot):
        if self.staging.pop(expert) != slot:
            raise RuntimeError('Refinement slot was not reserved for this expert')
        self.available.append(slot)

    def copy_from(self, other, expert, wanted):
        """Install an expert's refinement from another bank of the same layer, without an SSD read."""
        slot = other.slots[expert]
        payload = {name: {field: storage[slot] for field, storage in fields.items()}
                   for name, fields in other.host.items()}
        return self.install(expert, payload, wanted)

    def device_table(self):
        # Rebuilt only after an install changes the slot map. Installs run between tokens, so no queued
        # GPU work can observe a partially updated table.
        if self._device_table is None:
            self._device_table = mx.array(self.table)
        return self._device_table

    def retain(self, output):
        self.previous_output = output

    def finish(self):
        if self.previous_output is not None:
            mx.eval(self.previous_output)
            self.previous_output = None
