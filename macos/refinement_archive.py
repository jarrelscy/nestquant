"""Expert-major 4-bit refinement archive: one contiguous, 16 KiB-aligned record per (layer, expert).

The checkpoint stores each expert's refinement as ~40 tensors spread over eight tensor-parallel shard files.
RefinementReader.read gathers them with many small reads, then concatenates, validates and copies them
under the GIL. This archive stores each refinement exactly as PackedRefinementBank holds it in a slot:
per projection (gate, up, down) the planes p4, word, suh4, svh4 and lrU4, with lrU4 zero padded to the
bank's rank for the layer (the rank of the resident base's lrV plane). A record is read with one os.preadv
(the GIL is released for the whole read) into a pooled buffer, a prefill staging row, or a bank slot.

Layout (version 1)
  <archive>/index.json        format, layout version, checkpoint revision, per-layer file name, size, SHA-256,
                              record stride and field table (projection, field, dtype, shape, offset, bytes)
  <archive>/L{n}.bin          288 records; record e starts at e * stride (stride is a multiple of 16 KiB),
                              fields are packed in table order at 64-byte aligned offsets, padding is zero
  <archive>/L{n}.json         per-layer evidence: record SHA-256 for every expert, source fingerprints

Conversion is resumable (per-record progress with digests), builds every record from the current
RefinementReader.read output with the bank's padding rule, and finishes each layer by reading the whole file
back and checking every record digest before the layer is published.

Usage
  python macos/refinement_archive.py convert --model <checkpoint> --base <resident-banks-v1> --output <dir>
  python macos/refinement_archive.py verify --model <checkpoint> --output <dir> [--sample 64]
"""

from collections import deque
import fcntl
import hashlib
import json
import math
import mmap
import os
from pathlib import Path
import shutil
import struct
import threading
import time
import weakref

import numpy as np

FORMAT = 'nestquant-refinement-expert-major-v1'
LAYOUT_VERSION = 1
REVISION = '4023870a6f2c6097e37fa5d2e54040db64b011e9'
ALIGNMENT = 16384
FIELD_ALIGNMENT = 64
EXPERTS = 288
LAYERS = tuple(range(3, 45))
PROJECTIONS = ('gate', 'up', 'down')
FIELDS = ('p4', 'word', 'suh4', 'svh4', 'lrU4')
DTYPES = {'u1': np.dtype('u1'), '<i4': np.dtype('<i4'), '<f2': np.dtype('<f2')}
MAX_RECORD_BYTES = 32 * 1024**2
F_RDAHEAD = getattr(fcntl, 'F_RDAHEAD', 45)  # macOS <sys/fcntl.h>
F_NOCACHE = fcntl.F_NOCACHE


def _align(value, alignment):
    return -(-value // alignment) * alignment


def layer_fields(proj_meta, ranks):
    """Field table for one layer. `ranks` maps projection -> bank low-rank rows (0: no lrU4 plane)."""
    fields, offset = [], 0
    for projection in PROJECTIONS:
        meta = proj_meta[projection]
        units = meta['tk'] * meta['tn']
        shapes = [('p4', 'u1', (units * int(meta['res_rule']['K'] * 256),)), ('word', '<i4', (units,)),
                  ('suh4', '<f2', (meta['k'],)), ('svh4', '<f2', (meta['n'],))]
        if ranks[projection]:
            shapes.append(('lrU4', '<f2', (ranks[projection], meta['n'])))
        for field, dtype, shape in shapes:
            offset = _align(offset, FIELD_ALIGNMENT)
            size = math.prod(shape) * DTYPES[dtype].itemsize
            fields.append(dict(projection=projection, field=field, dtype=dtype, shape=list(shape),
                               offset=offset, bytes=size))
            offset += size
    return fields, offset, _align(offset, ALIGNMENT)


def validate_fields(fields, record_bytes, stride):
    """Check a field table read from an index: known names and dtypes, ordered, aligned, nonoverlapping."""
    if not isinstance(fields, list) or not fields:
        raise ValueError('Missing refinement field table')
    position, seen = 0, set()
    for item in fields:
        if not isinstance(item, dict) or set(item) != {'projection', 'field', 'dtype', 'shape', 'offset', 'bytes'}:
            raise ValueError('Malformed refinement field')
        key = item['projection'], item['field']
        if (item['projection'] not in PROJECTIONS or item['field'] not in FIELDS or key in seen
                or item['dtype'] not in DTYPES or not isinstance(item['shape'], list) or not item['shape']
                or any(type(n) is not int or n <= 0 for n in item['shape'])):
            raise ValueError(f'Invalid refinement field {key}')
        seen.add(key)
        size = math.prod(item['shape']) * DTYPES[item['dtype']].itemsize
        if (type(item['offset']) is not int or item['offset'] < position or item['offset'] % FIELD_ALIGNMENT
                or item['bytes'] != size):
            raise ValueError(f'Refinement field {key} overlaps or is misaligned')
        position = item['offset'] + size
    if position != record_bytes or stride % ALIGNMENT or not record_bytes <= stride <= MAX_RECORD_BYTES:
        raise ValueError('Refinement record size or stride is invalid')
    for projection in PROJECTIONS:
        if not {(projection, f) for f in FIELDS[:4]} <= seen:
            raise ValueError(f'Missing {projection} refinement planes')


def ranks_from_manifest(manifest):
    return {p: max(int(manifest['per_expert'][str(e)]['lr_rank'][p]) for e in range(EXPERTS)) for p in PROJECTIONS}


def ranks_from_base(base_dir, layer):
    """The bank pads lrU4 to the resident base's lrV rank; read it from the base archive header."""
    path = Path(base_dir) / f'L{layer}.safetensors'
    with path.open('rb') as handle:
        length = struct.unpack('<Q', handle.read(8))[0]
        if not 2 <= length <= 16 * 1024**2:
            raise ValueError('Invalid base archive header')
        header = json.loads(handle.read(length))
    meta = header.get('__metadata__', {})
    if (meta.get('revision'), meta.get('layer')) != (REVISION, str(layer)):
        raise ValueError(f'Unexpected base archive metadata for layer {layer}')
    ranks = {}
    for projection in PROJECTIONS:
        info = header.get(f'{projection}.lrV') or (header.get('gate.lrV') if projection == 'up' else None)
        ranks[projection] = int(info['shape'][1]) if info else 0
    return ranks


def slot_image(payload, fields, out):
    """Write a RefinementReader payload into a record buffer with PackedRefinementBank.write's rules."""
    by_projection = {}
    for item in fields:
        by_projection.setdefault(item['projection'], set()).add(item['field'])
    for projection, names in by_projection.items():
        if set(payload[projection]) - names:
            raise ValueError('Unexpected refinement plane')
    for item in fields:
        target = out[item['offset']:item['offset'] + item['bytes']].view(DTYPES[item['dtype']]).reshape(item['shape'])
        value = payload[item['projection']].get(item['field'])
        if item['field'] == 'lrU4':
            target.fill(0)
            if value is not None:
                if value.ndim != 2 or value.shape[0] > target.shape[0] or value.shape[1] != target.shape[1]:
                    raise ValueError('Unexpected low-rank refinement shape')
                np.copyto(target[:value.shape[0]], value, casting='no')
        elif value is None or value.shape != target.shape:
            raise ValueError('Missing or malformed refinement plane')
        else:
            np.copyto(target, value, casting='no')


def read_fully(fd, buffers, offset):
    """One vectored positional read (more only if the kernel returns short). The GIL is released in preadv."""
    views = [memoryview(b).cast('B') for b in buffers]
    try:
        total = sum(v.nbytes for v in views)
        completed, index, skipped = 0, 0, 0
        while index < len(views):
            first = views[index][skipped:]
            try:
                count = os.preadv(fd, [first, *views[index + 1:]], offset + completed)
            finally:
                first.release()
            if count <= 0:
                raise EOFError('Refinement record ended during positional read')
            completed += count
            while index < len(views) and count >= views[index].nbytes - skipped:
                count -= views[index].nbytes - skipped
                index += 1
                skipped = 0
            skipped += count
        return total
    finally:
        for view in views:
            view.release()


def fingerprint(path):
    stat = Path(path).stat()
    return dict(bytes=stat.st_size, inode=stat.st_ino, device=stat.st_dev, mtime_ns=stat.st_mtime_ns)


def source_state(model, layer):
    directory = Path(model) / 'layers' / f'L{layer}'
    manifest = (directory / 'manifest.json').read_bytes()
    shards = {f'tp{s}': {k: v for k, v in fingerprint(directory / f'tp{s}.safetensors').items()
                         if k in ('bytes', 'mtime_ns')} for s in range(8)}
    return dict(manifest_sha256=hashlib.sha256(manifest).hexdigest(), shards=shards)


def _env_flag(name, default):
    value = os.environ.get(name)
    return default if value is None or value == '' else value not in ('0', 'false', 'no', 'off')


class _BufferPool:
    """Page-aligned record buffers returned automatically once every NumPy view of a read is gone."""

    def __init__(self, size, keep=32):
        self.size, self.keep = size, keep
        self.free = deque()
        self.allocated = 0

    def take(self):
        try:
            mapping = self.free.pop()
        except IndexError:
            mapping = mmap.mmap(-1, self.size)
            self.allocated += 1
        owner = np.frombuffer(mapping, dtype=np.uint8)
        weakref.finalize(owner, self._give, mapping)
        return owner

    def _give(self, mapping):
        if len(self.free) < self.keep:
            self.free.append(mapping)


class ExpertMajorRefinementReader:
    """Drop-in for RefinementReader backed by an expert-major archive.

    read(layer, expert) -> (payload, bytes_read, seconds), payload = {projection: {field: ndarray}} with
    NumPy views (no copies) into a pooled buffer; PackedRefinementBank.install/write accept it directly.
    Thread safe: all layer files are opened up front and positional reads share no state.

    cache=False sets F_NOCACHE (today's behaviour for shard reads); cache=True lets the unified buffer cache
    keep recently read records, so a re-fetch after eviction can come from RAM. readahead=False sets
    F_RDAHEAD 0 (records are read whole, so speculative read-ahead only adds unrelated I/O).
    """

    thread_safe = True

    def __init__(self, archive, root=None, cache=None, readahead=None, check=True, verify=True, layers=LAYERS):
        self.archive = Path(archive)
        self.root = Path(root) if root is not None else None
        self.cache = _env_flag('NQ_REFINEMENT_CACHE', False) if cache is None else bool(cache)
        self.readahead = _env_flag('NQ_REFINEMENT_READAHEAD', True) if readahead is None else bool(readahead)
        self.verify = verify
        self.lock = threading.Lock()
        self.manifests = {}
        self.geometry = None
        self.fds = {}
        index = json.loads((self.archive / 'index.json').read_text())
        if (index.get('format'), index.get('layout_version'), index.get('revision')) != (FORMAT, LAYOUT_VERSION, REVISION):
            raise ValueError('Unsupported refinement archive')
        if index.get('alignment') != ALIGNMENT or index.get('experts') != EXPERTS:
            raise ValueError('Unexpected refinement archive geometry')
        self.layers = {}
        try:
            for key, entry in index['layers'].items():
                layer = int(key)
                if layer not in LAYERS:
                    raise ValueError('Invalid refinement archive layer')
                validate_fields(entry['fields'], entry['record_bytes'], entry['stride'])
                path = self.archive / entry['file']
                if path.parent != self.archive:
                    raise ValueError('Refinement archive files must live in the archive directory')
                state = fingerprint(path)
                if state['bytes'] != EXPERTS * entry['stride'] or (check and state != entry['state']):
                    raise RuntimeError(f'Refinement archive changed: {path.name}')
                fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
                self.fds[layer] = fd
                fcntl.fcntl(fd, F_NOCACHE, 0 if self.cache else 1)
                if not self.readahead:
                    fcntl.fcntl(fd, F_RDAHEAD, 0)
                fields = [dict(item, np_dtype=DTYPES[item['dtype']], shape=tuple(item['shape'])) for item in entry['fields']]
                self.layers[layer] = dict(fields=fields, record_bytes=entry['record_bytes'], stride=entry['stride'],
                                          ranks=entry['ranks'], fd=fd)
            if set(self.layers) != set(layers):
                raise ValueError('Refinement archive is incomplete')
        except BaseException:
            self.close()
            raise
        largest = max(entry['record_bytes'] for entry in self.layers.values())
        self.pool = _BufferPool(_align(largest, ALIGNMENT))
        self.scratch = bytearray(FIELD_ALIGNMENT)

    # Compatibility with RefinementReader / ShardedRefinementReader callers.
    def _manifest(self, layer):
        with self.lock:
            if layer not in self.manifests:
                if self.root is None:
                    raise RuntimeError('Pass the checkpoint root to read manifests')
                self.manifests[layer] = json.loads((self.root / 'layers' / f'L{layer}' / 'manifest.json').read_text())
            return self.manifests[layer]

    def manifest(self, layer):
        manifest = self._manifest(layer)
        if self.geometry is None:
            from prefill_stream import LAYOUT_KEYS
            self.geometry = {name: {key: manifest['proj_meta'][name][key] for key in LAYOUT_KEYS} for name in PROJECTIONS}
        return manifest

    def bytes_for(self, layer, expert):
        return self.layers[layer]['record_bytes']

    def _iovecs(self, layer, targets):
        """Buffers covering [0, record_bytes) of a record: each field's target, scratch for alignment gaps."""
        buffers, position = [], 0
        for item, target in zip(self.layers[layer]['fields'], targets):
            gap = item['offset'] - position
            if gap:
                buffers.append(memoryview(self.scratch)[:gap])
            buffers.append(target)
            position = item['offset'] + item['bytes']
        return buffers

    def _check(self, layer, words):
        if self.verify:
            for word in words:
                if np.any((word & 255) == 0):
                    raise ValueError(f'Invalid zero denominator in refinement block (layer {layer})')

    def read_record(self, layer, expert):
        """Raw record bytes (one preadv into a pooled buffer). Returns (uint8 array, bytes, seconds)."""
        started = time.perf_counter()
        entry = self.layers[layer]
        if not 0 <= expert < EXPERTS:
            raise ValueError('Invalid expert')
        owner = self.pool.take()
        record = owner[:entry['record_bytes']]
        count = read_fully(entry['fd'], [record], expert * entry['stride'])
        return record, count, time.perf_counter() - started

    def read(self, layer, expert):
        started = time.perf_counter()
        record, count, _ = self.read_record(layer, expert)
        payload = {projection: {} for projection in PROJECTIONS}
        for item in self.layers[layer]['fields']:
            payload[item['projection']][item['field']] = (
                record[item['offset']:item['offset'] + item['bytes']].view(item['np_dtype']).reshape(item['shape']))
        self._check(layer, (planes['word'] for planes in payload.values()))
        return payload, count, time.perf_counter() - started

    def read_into(self, layer, expert, staging, slot):
        """Fill prefill staging row `slot` (prefill_stream.RefinementStaging). Returns (bytes, seconds)."""
        started = time.perf_counter()
        entry = self.layers[layer]
        targets = []
        for item in entry['fields']:
            row = staging.host[item['projection']][item['field']][slot]
            if item['field'] == 'lrU4':
                if item['shape'][0] > row.shape[0] or item['shape'][1] != row.shape[1]:
                    raise ValueError('Refinement low-rank plane exceeds the staging rank')
                row = row[:item['shape'][0]]
            if row.nbytes != item['bytes'] or row.dtype != item['np_dtype'] or not row.flags.c_contiguous:
                raise ValueError('Staging row does not match the refinement archive')
            targets.append(row)
        count = read_fully(entry['fd'], self._iovecs(layer, targets), expert * entry['stride'])
        for projection in PROJECTIONS:
            rank = entry['ranks'][projection]
            staging.host[projection]['lrU4'][slot][rank:] = 0
        self._check(layer, (staging.host[p]['word'][slot] for p in PROJECTIONS))
        return count, time.perf_counter() - started

    def slot_targets(self, layer, bank, slot):
        """Bank slot views in record order, or None if the bank's planes do not match the record exactly."""
        entry = self.layers[layer]
        targets = []
        expected = {(item['projection'], item['field']) for item in entry['fields']}
        present = {(p, f) for p, planes in bank.host.items() for f in planes}
        if expected != present:
            return None
        for item in entry['fields']:
            target = bank.host[item['projection']][item['field']][slot]
            if target.nbytes != item['bytes'] or target.dtype != item['np_dtype'] or not target.flags.c_contiguous:
                return None
            targets.append(target)
        return targets

    def read_into_slot(self, layer, expert, bank, slot):
        """Read a record straight into a reserved PackedRefinementBank slot (no intermediate buffer).

        The slot must be reserved (out of the slot table), exactly like PackedRefinementBank.write."""
        started = time.perf_counter()
        targets = self.slot_targets(layer, bank, slot)
        if targets is None:
            payload, count, _ = self.read(layer, expert)
            bank.write(slot, payload)
            return count, time.perf_counter() - started
        count = read_fully(self.layers[layer]['fd'], self._iovecs(layer, targets),
                           expert * self.layers[layer]['stride'])
        self._check(layer, (bank.host[p]['word'][slot] for p in PROJECTIONS))
        return count, time.perf_counter() - started

    def close(self):
        fds, self.fds = self.fds, {}
        for fd in fds.values():
            os.close(fd)


DEFAULT_ARCHIVE = Path.home() / 'Models/GLM/refinements-expert-major-v1'


def reader_from_env(root):
    """ExpertMajorRefinementReader for NQ_REFINEMENT_ARCHIVE, or for the default archive path when it exists;
    None (use the shard reader) when neither is available or NQ_REFINEMENT_ARCHIVE is 0/off."""
    archive = os.environ.get('NQ_REFINEMENT_ARCHIVE')
    if archive in ('0', 'off', 'none'):
        return None
    if not archive:
        if not (DEFAULT_ARCHIVE / 'index.json').exists():
            return None
        archive = str(DEFAULT_ARCHIVE)
    return ExpertMajorRefinementReader(Path(archive).expanduser(), root=root)


# ----------------------------------------------------------------------------------------------- conversion

def _layer_paths(output, layer):
    return output / f'L{layer}.bin', output / f'L{layer}.json', output / f'L{layer}.progress'


def plan_layer(model, base, layer):
    manifest = json.loads((Path(model) / 'layers' / f'L{layer}' / 'manifest.json').read_text())
    needed = ranks_from_manifest(manifest)
    ranks = ranks_from_base(base, layer) if base is not None else needed
    if any(ranks[p] < needed[p] for p in PROJECTIONS):
        raise ValueError(f'Bank rank below the checkpoint rank in layer {layer}: {ranks} < {needed}')
    fields, record_bytes, stride = layer_fields(manifest['proj_meta'], ranks)
    return dict(fields=fields, record_bytes=record_bytes, stride=stride, ranks=ranks)


def _load_progress(path, header):
    done = {}
    if not path.exists():
        return done
    lines = path.read_text().splitlines()
    if not lines or json.loads(lines[0]) != header:
        raise RuntimeError(f'{path.name} belongs to a different source or layout; remove it to restart the layer')
    for line in lines[1:]:
        parts = line.split()
        if len(parts) == 2 and parts[0].isdigit() and len(parts[1]) == 64:
            done[int(parts[0])] = parts[1]
    return done


def convert_layer(model, base, output, layer, log=print, sync_every=16):
    from experts import RefinementReader

    model, output = Path(model), Path(output)
    target, evidence, progress = _layer_paths(output, layer)
    if evidence.exists():
        data = json.loads(evidence.read_text())
        if data.get('state') == fingerprint(target):
            return data
        raise RuntimeError(f'{target.name} changed after conversion; remove it and its evidence to redo the layer')
    plan = plan_layer(model, base, layer)
    source = source_state(model, layer)
    header = dict(format=FORMAT, layout_version=LAYOUT_VERSION, revision=REVISION, layer=layer, source=source,
                  fields=plan['fields'], stride=plan['stride'])
    done = _load_progress(progress, header)
    size = EXPERTS * plan['stride']
    if not target.exists():
        done = {}
        if shutil.disk_usage(output).free < 100 * 1024**3 + size:
            raise RuntimeError('Conversion must preserve a 100 GiB disk reserve')
        progress.write_text(json.dumps(header) + '\n')
        with target.open('xb') as handle:
            handle.truncate(size)
    if not progress.exists():
        done = {}
        progress.write_text(json.dumps(header) + '\n')
    if os.stat(target).st_size != size:
        raise RuntimeError(f'{target.name} has an unexpected size')
    reader = RefinementReader(model)
    started = time.perf_counter()
    image = np.zeros(plan['stride'], dtype=np.uint8)
    fd = os.open(target, os.O_RDWR | os.O_CLOEXEC)
    try:
        fcntl.fcntl(fd, F_NOCACHE, 1)
        for attempt in range(2):
            pending = []
            with progress.open('a') as log_file:
                for expert in range(EXPERTS):
                    if expert in done:
                        continue
                    payload, _, _ = reader.read(layer, expert)
                    image.fill(0)
                    slot_image(payload, plan['fields'], image)
                    del payload
                    digest = hashlib.sha256(image).hexdigest()
                    written = os.pwrite(fd, image, expert * plan['stride'])
                    if written != plan['stride']:
                        raise OSError('Short refinement record write')
                    pending.append((expert, digest))
                    if len(pending) >= sync_every or expert == EXPERTS - 1:
                        os.fsync(fd)
                        log_file.write(''.join(f'{e} {d}\n' for e, d in pending))
                        log_file.flush()
                        os.fsync(log_file.fileno())
                        done.update(pending)
                        pending = []
                if pending:
                    os.fsync(fd)
                    log_file.write(''.join(f'{e} {d}\n' for e, d in pending))
                    log_file.flush()
                    done.update(pending)
            # Read every record back from disk and check it against the digest of the RefinementReader output.
            bad, file_digest = [], hashlib.sha256()
            for expert in range(EXPERTS):
                read_fully(fd, [image], expert * plan['stride'])
                file_digest.update(image)
                if hashlib.sha256(image).hexdigest() != done[expert]:
                    bad.append(expert)
            if not bad:
                break
            log(json.dumps(dict(event='refinement_records_mismatch', layer=layer, experts=bad)))
            for expert in bad:
                del done[expert]
        else:
            raise RuntimeError(f'Layer {layer} records do not verify after rewrite')
    finally:
        os.close(fd)
        reader.close()
    if source_state(model, layer) != source:
        raise RuntimeError(f'Checkpoint layer {layer} changed during conversion')
    data = dict(header, record_bytes=plan['record_bytes'], ranks=plan['ranks'], file=target.name,
                sha256=file_digest.hexdigest(), record_sha256=[done[e] for e in range(EXPERTS)],
                state=fingerprint(target), seconds=time.perf_counter() - started)
    stage = evidence.with_suffix('.tmp')
    stage.write_text(json.dumps(data, indent=1) + '\n')
    stage.replace(evidence)
    progress.unlink()
    log(json.dumps(dict(event='refinement_layer_written', layer=layer, bytes=size, seconds=round(data['seconds'], 1))))
    return data


def write_index(output):
    output = Path(output)
    layers = {}
    for layer in LAYERS:
        target, evidence, _ = _layer_paths(output, layer)
        data = json.loads(evidence.read_text())
        if data['state'] != fingerprint(target):
            raise RuntimeError(f'{target.name} changed after conversion')
        layers[str(layer)] = {k: data[k] for k in ('file', 'sha256', 'stride', 'record_bytes', 'ranks', 'fields',
                                                   'state', 'source')}
        layers[str(layer)]['file_bytes'] = data['state']['bytes']
    index = dict(format=FORMAT, layout_version=LAYOUT_VERSION, revision=REVISION, alignment=ALIGNMENT,
                 experts=EXPERTS, layers=layers)
    stage = output / 'index.json.tmp'
    stage.write_text(json.dumps(index, indent=1) + '\n')
    stage.replace(output / 'index.json')
    return index


def _convert_worker(args):
    model, base, output, layer = args
    data = convert_layer(model, base, output, layer, log=lambda line: print(line, flush=True))
    return layer, data['seconds']


def convert(model, base, output, layers=LAYERS, workers=2):
    from concurrent.futures import ProcessPoolExecutor

    model_path = Path(model)
    if not (model_path / 'COMPLETE').exists():
        raise RuntimeError('Checkpoint is incomplete')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    jobs = [(str(model), str(base) if base else None, str(output), layer) for layer in layers]
    if workers <= 1:
        for job in jobs:
            _convert_worker(job)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for layer, seconds in executor.map(_convert_worker, jobs):
                pass
    if all(_layer_paths(output, layer)[1].exists() for layer in LAYERS):
        write_index(output)
    return time.perf_counter() - started


def verify(model, output, sample=None, seed=0):
    """Compare the archive reader with RefinementReader after the bank's padding rule (all or sampled experts)."""
    from experts import RefinementReader

    old = RefinementReader(Path(model))
    new = ExpertMajorRefinementReader(output, root=model)
    keys = [(layer, expert) for layer in LAYERS for expert in range(EXPERTS)]
    if sample:
        rng = np.random.default_rng(seed)
        keys = [keys[i] for i in sorted(rng.choice(len(keys), size=sample, replace=False))]
    try:
        for layer, expert in keys:
            entry = new.layers[layer]
            expected = np.zeros(entry['record_bytes'], dtype=np.uint8)
            slot_image(old.read(layer, expert)[0], entry['fields'], expected)
            payload, count, _ = new.read(layer, expert)
            actual = np.zeros_like(expected)
            slot_image(payload, entry['fields'], actual)
            if count != entry['record_bytes'] or not np.array_equal(actual, expected):
                raise AssertionError(f'Refinement record differs at layer {layer} expert {expert}')
    finally:
        old.close()
        new.close()
    return len(keys)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=('convert', 'verify', 'index'))
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--base', type=Path, help='Resident base archive (sets the bank lrU4 rank per layer)')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--layers', default=None, help='Comma list (default: all routed layers)')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--sample', type=int, default=None)
    args = parser.parse_args()
    if args.command == 'convert':
        layers = tuple(int(v) for v in args.layers.split(',')) if args.layers else LAYERS
        seconds = convert(args.model, args.base, args.output, layers, args.workers)
        print(json.dumps(dict(event='refinement_archive_done', seconds=round(seconds, 1))), flush=True)
    elif args.command == 'index':
        write_index(args.output)
    else:
        started = time.perf_counter()
        count = verify(args.model, args.output, args.sample)
        print(json.dumps(dict(event='refinement_archive_verified', records=count,
                              seconds=round(time.perf_counter() - started, 1))), flush=True)
