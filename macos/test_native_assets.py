import json
from pathlib import Path
import tempfile
import unittest

import mlx.core as mx

from native_assets import FORMAT, REVISION, fingerprint, paths, read_bank


class NativeAssetsTest(unittest.TestCase):
    def test_uncached_file_reader_preserves_arrays_after_close(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            file, sidecar = paths(root, 3)
            expected = {'base': mx.arange(65536, dtype=mx.uint8),
                        'suh2': mx.arange(256).astype(mx.float16),
                        'word': mx.arange(256, dtype=mx.int32)}
            geometry = {'gate': {'tk': 1, 'tn': 2}}
            mx.save_safetensors(file, {f'gate.{key}': value for key, value in expected.items()}, metadata={
                'format': FORMAT, 'revision': REVISION, 'layer': '3', 'geometry': json.dumps(geometry)})
            sidecar.write_text(json.dumps({'revision': REVISION, 'state': fingerprint(file)}))
            ordinary = read_bank(root, 3, 'mlx')
            uncached = read_bank(root, 3, 'mlx-nocache')
            file.unlink()
            self.assertEqual(uncached['gate'][1], geometry['gate'])
            for name, value in expected.items():
                self.assertEqual(uncached['gate'][0][name].dtype, value.dtype)
                self.assertTrue(mx.array_equal(uncached['gate'][0][name], value).item())
                self.assertTrue(mx.array_equal(uncached['gate'][0][name], ordinary['gate'][0][name]).item())


if __name__ == '__main__':
    unittest.main()
