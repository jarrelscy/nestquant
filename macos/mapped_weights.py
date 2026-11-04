import json
import math
import os
from pathlib import Path
import struct

import mlx.core as mx
import numpy as np


DTYPES = {
    'U8': mx.uint8,
    'I32': mx.int32,
    'U32': mx.uint32,
    'F16': mx.float16,
    'BF16': mx.bfloat16,
    'F32': mx.float32,
}


def read_layouts(handle, file_size):
    prefix = handle.read(8)
    if len(prefix) != 8:
        raise ValueError('Missing Safetensors header length')
    size, = struct.unpack('<Q', prefix)
    if not 0 < size <= 4 * 1024**2:
        raise ValueError('Unsupported Safetensors header size')
    header = json.loads(handle.read(size))
    if not isinstance(header, dict):
        raise ValueError('Safetensors header must be a JSON object')
    offset = 8 + size
    metadata = header.pop('__metadata__', {})
    if not isinstance(metadata, dict):
        raise ValueError('Safetensors metadata must be a JSON object')
    layouts = {}
    for name, entry in header.items():
        try:
            dtype = DTYPES.get(entry['dtype'])
            shape = tuple(entry['shape'])
            start, end = entry['data_offsets']
        except (KeyError, TypeError, ValueError):
            raise ValueError(f'Malformed Safetensors tensor entry: {name}') from None
        if (dtype is None or any(type(n) is not int or n < 0 for n in shape)
                or type(start) is not int or type(end) is not int
                or not 0 <= start <= end <= file_size - offset
                or end - start != math.prod(shape) * dtype.size):
            raise ValueError(f'Unsupported mapped Safetensors layout: {name}')
        layouts[name] = (start, end, dtype, shape)
    previous = 0
    for start, end, _, _ in sorted(layouts.values(), key=lambda entry: entry[:2]):
        if start != previous:
            raise ValueError('Safetensors data offsets overlap or leave gaps')
        previous = end
    if previous != file_size - offset:
        raise ValueError('Unexpected Safetensors payload length')
    return layouts, metadata, offset


def load_mapped(path: Path):
    with path.open('rb') as handle:
        layouts, metadata, offset = read_layouts(handle, os.fstat(handle.fileno()).st_size)
        mapping = np.memmap(handle, dtype=np.uint8, mode='c')
    # The DLPack capsule retains the private mapping. MLX must import it without a copy.
    buffer = mx.from_dlpack(mapping, copy=False)
    weights = {}
    copied = 0
    mapped = 0
    for name, (start, end, dtype, shape) in layouts.items():
        start += offset
        end += offset
        if start % dtype.size:
            copied += end - start
            if copied > 48 * 1024**2:
                raise ValueError(f'Unaligned weights exceed the bounded copy size: {name}')
            raw = mx.array(mapping[start:end])
        else:
            raw = buffer[start:end]
            mapped += 1
        weights[name] = raw.view(dtype).reshape(shape)
    if mapped < 2:
        raise ValueError('Expected at least two views sharing the mapped storage')
    return weights, metadata
