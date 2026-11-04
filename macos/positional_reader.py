import fcntl
import json
import math
import os

import numpy as np


class PositionalTensorReader:
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

    def get_tensor(self, name):
        if self.fd is None:
            raise RuntimeError('Tensor reader is closed')
        info = self.header[name]
        if not isinstance(info, dict) or info.get('dtype') not in self.DTYPES:
            raise ValueError('Unsupported refinement dtype')
        dtype = self.DTYPES[info['dtype']]
        shape, offsets = info.get('shape'), info.get('data_offsets')
        if not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape):
            raise ValueError('Invalid tensor shape')
        if not isinstance(offsets, list) or len(offsets) != 2 or any(type(n) is not int for n in offsets):
            raise ValueError('Invalid tensor offsets')
        begin, end = offsets
        size = math.prod(shape) * dtype.itemsize
        if not 0 <= begin <= end <= self.file_size - self.data_start or end - begin != size or size > 64 * 1024**2:
            raise ValueError('Tensor range or byte count is invalid')
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
        return np.frombuffer(storage, dtype=dtype).reshape(shape)

    def close(self):
        if self.fd is not None:
            fd, self.fd = self.fd, None
            os.close(fd)
