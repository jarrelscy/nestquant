import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct

from mapped_weights import DTYPES, read_layouts
from native_assets import REVISION, fingerprint, paths


FORMAT = 'nestquant-expert-major-base-v1'
MAX_ROW_BYTES = 16 * 1024**2


def read_into(fd, buffers, offset):
    views = []
    try:
        views = [memoryview(buffer).cast('B') for buffer in buffers]
        if not views or any(not view.nbytes or view.readonly for view in views):
            raise ValueError('Expected nonempty writable buffers')
        index, skipped, completed = 0, 0, 0
        while index < len(views):
            first = views[index][skipped:]
            try:
                count = os.preadv(fd, [first, *views[index + 1:]], offset + completed)
            finally:
                first.release()
            if count <= 0:
                raise EOFError('Expert row ended during positional read')
            completed += count
            while index < len(views) and count >= views[index].nbytes - skipped:
                count -= views[index].nbytes - skipped
                index += 1
                skipped = 0
            skipped += count
        return completed
    finally:
        for view in views:
            view.release()


def plane_layouts(layouts, metadata):
    if metadata.get('format') != FORMAT or set(layouts) != {'experts'}:
        raise ValueError('Unsupported expert-major archive')
    start, end, dtype, shape = layouts['experts']
    if (dtype != DTYPES['U8'] or len(shape) != 2 or shape[0] != 288
            or not 0 < shape[1] <= MAX_ROW_BYTES or start != 0 or end != 288 * shape[1]):
        raise ValueError('Invalid expert-major row storage')
    descriptions = json.loads(metadata['plane_layouts'])
    if not isinstance(descriptions, list) or not descriptions:
        raise ValueError('Missing expert-major planes')
    result, position = {}, 0
    for item in descriptions:
        if not isinstance(item, dict) or not {'name', 'dtype', 'shape', 'offset'} <= item.keys():
            raise ValueError('Malformed expert-major plane description')
        name, tag, dimensions = item['name'], item['dtype'], item['shape']
        if (not isinstance(name, str) or len(name.split('.')) != 2 or name in result
                or not isinstance(tag, str) or tag not in DTYPES or not isinstance(dimensions, list)
                or any(type(value) is not int or value <= 0 for value in dimensions)):
            raise ValueError('Invalid expert-major plane')
        dtype = DTYPES[tag]
        size = math.prod(dimensions) * dtype.size
        if type(item['offset']) is not int or item['offset'] != position or position + size > shape[1]:
            raise ValueError('Expert-major planes overlap or leave gaps')
        result[name] = position, position + size, dtype, (288, *dimensions)
        position += size
    if position != shape[1]:
        raise ValueError('Expert-major planes do not cover the row')
    return result, shape[1]


def convert_layer(source, destination, layer, check):
    file, sidecar = paths(source, layer)
    target, proof = paths(destination, layer)
    evidence = json.loads(sidecar.read_text())
    if evidence['revision'] != REVISION or evidence['state'] != fingerprint(file):
        raise RuntimeError('Source packed archive changed')
    if target.exists() or proof.exists():
        raise FileExistsError('Use a fresh expert-major archive destination')
    temporary = target.with_suffix('.tmp.safetensors')
    with file.open('rb', buffering=0) as reader:
        fcntl.fcntl(reader, fcntl.F_NOCACHE, 1)
        layouts, metadata, data_start = read_layouts(reader, os.fstat(reader.fileno()).st_size)
        if (metadata.get('format'), metadata.get('revision'), metadata.get('layer')) != (
                'nestquant-resident-base-v1', REVISION, str(layer)):
            raise ValueError('Unexpected source base archive')
        names, row_bytes = [], 0
        for name, (start, end, dtype, shape) in layouts.items():
            if not shape or shape[0] != 288:
                raise ValueError('Expected 288 source experts')
            size = (end - start) // 288
            names.append(dict(name=name, dtype=next(tag for tag, value in DTYPES.items() if value == dtype),
                              shape=list(shape[1:]), offset=row_bytes))
            row_bytes += size
        if not 0 < row_bytes <= MAX_ROW_BYTES:
            raise ValueError('Expert row exceeds the bounded preparation buffer')
        metadata = dict(metadata, format=FORMAT, plane_layouts=json.dumps(names, separators=(',', ':')))
        header = dict(__metadata__=metadata, experts=dict(dtype='U8', shape=[288, row_bytes],
                                                        data_offsets=[0, 288 * row_bytes]))
        encoded = json.dumps(header, separators=(',', ':')).encode()
        encoded += b' ' * (-len(encoded) % 8)
        prefix = struct.pack('<Q', len(encoded)) + encoded
        if shutil.disk_usage(destination).free < 100 * 1024**3 + len(prefix) + 288 * row_bytes:
            raise RuntimeError('Expert-major preparation must preserve the 100 GiB disk reserve')
        check()
        digest = hashlib.sha256(prefix)
        with temporary.open('xb', buffering=0) as writer:
            fcntl.fcntl(writer, fcntl.F_NOCACHE, 1)
            if writer.write(prefix) != len(prefix):
                raise OSError('Incomplete expert-major header write')
            payload = bytearray(row_bytes)
            for expert in range(288):
                check()
                for item in names:
                    start, end, _, _ = layouts[item['name']]
                    size = (end - start) // 288
                    with memoryview(payload)[item['offset']:item['offset'] + size] as view:
                        read_into(reader.fileno(), [view], data_start + start + expert * size)
                if writer.write(payload) != row_bytes:
                    raise OSError('Incomplete expert-major row write')
                digest.update(payload)
            os.fsync(writer.fileno())
    if evidence['state'] != fingerprint(file):
        raise RuntimeError('Source archive changed during preparation')
    check()
    os.link(temporary, target)
    temporary.unlink()
    with proof.open('x') as output:
        json.dump(dict(revision=REVISION, state=fingerprint(target), sha256=digest.hexdigest(),
                       source_sha256=evidence['sha256']), output, indent=2)
        output.write('\n')
    return dict(layer=layer, bytes=target.stat().st_size, row_bytes=row_bytes, planes=len(names))


if __name__ == '__main__':
    import argparse
    from metrics import MemoryGuard

    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--layers', type=int, nargs='+', choices=range(3, 45), default=list(range(3, 45)))
    args = parser.parse_args()
    args.output_dir.mkdir(exist_ok=True)
    guard = MemoryGuard(4)
    for layer in args.layers:
        result = convert_layer(args.source, args.output_dir, layer, lambda: guard.check(force=True))
        print(json.dumps(result), flush=True)
