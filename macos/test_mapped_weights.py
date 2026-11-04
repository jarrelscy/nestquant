import gc
import json
from pathlib import Path
import struct
import tempfile
import unittest

import mlx.core as mx

from mapped_weights import load_mapped


class MappedWeightsTest(unittest.TestCase):
    def test_mapped_and_unaligned_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'weights.safetensors'
            fields = {
                'codes': mx.arange(1, 28, dtype=mx.uint8).reshape(3, 9),
                'signs': mx.array([1, 0, 1], dtype=mx.uint8),
                'scales': mx.array([0.125, 1, -2], dtype=mx.float16),
                'word': mx.array([123, -456], dtype=mx.int32),
            }
            mx.save_safetensors(path, fields, metadata={'revision': 'fixture'})
            arrays, metadata = load_mapped(path)
            gc.collect()
            self.assertEqual(metadata, {'revision': 'fixture'})
            for name, array in arrays.items():
                self.assertEqual(array.dtype, fields[name].dtype)
                self.assertEqual(array.shape, fields[name].shape)
                self.assertTrue(mx.array_equal(array, fields[name]).item())

    def test_rejects_overlapping_offsets(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.safetensors'
            header = json.dumps({
                'a': {'dtype': 'U8', 'shape': [2], 'data_offsets': [0, 2]},
                'b': {'dtype': 'U8', 'shape': [2], 'data_offsets': [1, 3]},
            }).encode()
            path.write_bytes(struct.pack('<Q', len(header)) + header + bytes(3))
            with self.assertRaisesRegex(ValueError, 'overlap'):
                load_mapped(path)


if __name__ == '__main__':
    unittest.main()
