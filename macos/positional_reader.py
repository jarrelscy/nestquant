import fcntl
import json
import math
import os

import numpy as np


class PositionalTensorReader:
    MAX_RANGE_BYTES = 64 * 1024**2
    MAX_GAP_BYTES = 4096
    MAX_EXTRA_BYTES = 32 * 1024
    DTYPES = {'U8': np.dtype('u1'), 'I32': np.dtype('<i4'), 'F16': np.dtype('<f2')}

    def __init__(self, path):
        self.fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
        try:
            fcntl.fcntl(self.fd, fcntl.F_NOCACHE, 1)
            self.file_size = os.fstat(self.fd).st_size
            prefix = os.pread(self.fd, 8, 0)
            if len(prefix) != 8:
                raise ValueError('Truncated safetensors length')
            length = int.from_bytes(prefix, 'little')
            if not 2 <= length <= 16 * 1024**2 or length + 8 > self.file_size:
                raise ValueError('Invalid safetensors header length')
            header = os.pread(self.fd, length, 8)
            if len(header) != length:
                raise ValueError('Truncated safetensors header')
            self.header = json.loads(header)
            if not isinstance(self.header, dict) or self.header.get('__metadata__', {}).get('format') != 'nestquant-v1-tp-safetensors':
                raise ValueError('Unsupported NestQuant container')
            self.data_start = length + 8
        except BaseException:
            self.close()
            raise

    def _tensor_range(self, name):
        info = self.header[name]
        if not isinstance(info, dict) or info.get('dtype') not in self.DTYPES:
            raise ValueError('Unsupported expert tensor dtype')
        dtype = self.DTYPES[info['dtype']]
        shape, offsets = info.get('shape'), info.get('data_offsets')
        if not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape):
            raise ValueError('Invalid tensor shape')
        if not isinstance(offsets, list) or len(offsets) != 2 or any(type(n) is not int for n in offsets):
            raise ValueError('Invalid tensor offsets')
        begin, end = offsets
        size = math.prod(shape) * dtype.itemsize
        if not 0 <= begin <= end <= self.file_size - self.data_start or end - begin != size or size > self.MAX_RANGE_BYTES:
            raise ValueError('Tensor range or byte count is invalid')
        return begin, end, name, dtype, shape

    def get_tensor(self, name):
        return self.get_tensors((name,))[name]

    def get_tensors(self, names):
        if self.fd is None:
            raise RuntimeError('Tensor reader is closed')
        ranges = sorted(self._tensor_range(name) for name in dict.fromkeys(names))
        groups = []
        extra = 0
        for item in ranges:
            begin, end, *_ = item
            if groups:
                previous_end = groups[-1][-1][1]
                gap = begin - previous_end
                if gap < 0:
                    raise ValueError('Overlapping tensor ranges')
                if (gap <= self.MAX_GAP_BYTES and extra + gap <= self.MAX_EXTRA_BYTES
                        and end - groups[-1][0][0] <= self.MAX_RANGE_BYTES):
                    groups[-1].append(item)
                    extra += gap
                    continue
            groups.append([item])

        result = {}
        for group in groups:
            start = group[0][0]
            storage = self._read_range(start, group[-1][1] - start)
            for begin, end, name, dtype, shape in group:
                result[name] = np.frombuffer(storage, dtype=dtype, count=(end - begin) // dtype.itemsize,
                                             offset=begin - start).reshape(shape)
        return result

    def _read_range(self, begin, size):
        storage = bytearray(size)
        view = memoryview(storage)
        try:
            completed = 0
            while completed < size:
                # Python releases the GIL around preadv, unlike this safetensors version's NumPy pread path.
                count = os.preadv(self.fd, [view[completed:]], self.data_start + begin + completed)
                if count == 0:
                    raise EOFError('Tensor file ended during positional read')
                completed += count
        finally:
            view.release()
        return storage

    def close(self):
        if self.fd is not None:
            fd, self.fd = self.fd, None
            os.close(fd)
