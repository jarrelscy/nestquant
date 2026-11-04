from contextlib import closing
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mlx.core as mx
import numpy as np
from safetensors.numpy import save_file

from positional_reader import PositionalTensorReader
from experts import ExpertStore


class PositionalReaderTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'weights.safetensors'
        self.values = {'a': np.arange(32, dtype=np.float16).reshape(4, 8),
                       'b': np.arange(17, dtype=np.uint8), 'c': np.arange(9, dtype=np.int32)}
        save_file(self.values, self.path, metadata={'format': 'nestquant-v1-tp-safetensors'})

    def test_exact_owned_arrays_survive_close(self):
        reader = PositionalTensorReader(self.path)
        arrays = {name: reader.get_tensor(name) for name in self.values}
        reader.close()
        reader.close()
        for name, expected in self.values.items():
            np.testing.assert_array_equal(arrays[name], expected)
            arrays[name].fill(0)
        with self.assertRaises(RuntimeError):
            reader.get_tensor('a')

    def test_short_reads_fill_the_original_buffer(self):
        original = os.preadv
        with closing(PositionalTensorReader(self.path)) as reader:
            def partial(fd, buffers, offset):
                return original(fd, [buffers[0][:7]], offset)
            with patch('positional_reader.os.preadv', side_effect=partial):
                np.testing.assert_array_equal(reader.get_tensor('a'), self.values['a'])

    def test_eof_is_an_error(self):
        with closing(PositionalTensorReader(self.path)) as reader, patch('positional_reader.os.preadv', return_value=0):
            with self.assertRaises(EOFError):
                reader.get_tensor('a')

    def test_invalid_header_and_tensor_ranges_are_rejected(self):
        for info in ({'dtype': 'F16', 'shape': [4, 8], 'data_offsets': [0, 2]},
                     {'dtype': 'F16', 'shape': [-1], 'data_offsets': [0, 0]},
                     {'dtype': 'F16', 'shape': [True], 'data_offsets': [0, 2]},
                     {'dtype': 'F16', 'shape': [1], 'data_offsets': [-1, 1]},
                     {'dtype': 'F64', 'shape': [1], 'data_offsets': [0, 8]}):
            with self.subTest(info=info), closing(PositionalTensorReader(self.path)) as reader:
                reader.header['bad'] = info
                with self.assertRaises(ValueError):
                    reader.get_tensor('bad')
        self.path.write_bytes((2**63).to_bytes(8, 'little'))
        with self.assertRaises(ValueError):
            PositionalTensorReader(self.path)

    def test_synchronous_upgrade_preserves_shared_safetensors_handles(self):
        root = Path(self.directory.name)
        layer = root / 'layers/L3'
        layer.mkdir(parents=True)
        projections = ('gate', 'up', 'down')
        meta = dict(tk=1, tn=8, k=128, n=128, res_rule={'K': 2.5})
        manifest = {'proj_meta': {p: meta for p in projections},
                    'per_expert': {'0': {'lr_rank': {p: 0 for p in projections}}}}
        (layer / 'manifest.json').write_text(json.dumps(manifest))
        fields = {}
        for p in projections:
            fields[f'E0.{p}.p4'] = np.zeros(640, dtype=np.uint8)
            fields[f'E0.{p}.word'] = np.ones(1, dtype=np.int32)
            fields[f'E0.{p}.suh4'] = np.ones(16 if p == 'down' else 128, dtype=np.float16)
            fields[f'E0.{p}.svh4'] = np.ones(128 if p == 'down' else 16, dtype=np.float16)
        for shard in range(8):
            save_file(fields, layer / f'tp{shard}.safetensors', metadata={'format': 'nestquant-v1-tp-safetensors'})
        store = ExpertStore(root, cache_gib=0.01, resident_base=True)
        try:
            base = {p: ({'base': mx.zeros(1, dtype=mx.uint8)}, meta) for p in projections}
            store.base_cache[3, 0] = base
            store._handle(3, 0)
            result = store.get(3, 0, 4)
            for p in projections:
                self.assertEqual(result[p][0]['p4'].shape, (5120,))
                self.assertIs(result[p][0]['base'], base[p][0]['base'])
            self.assertEqual(len(store.handles), 8)
            self.assertTrue(all(isinstance(value, tuple) for value in store.handles.values()))
            self.assertIs(store.get(3, 0, 2), base)
        finally:
            store.close()
        self.assertFalse(store.handles)


if __name__ == '__main__':
    unittest.main()
