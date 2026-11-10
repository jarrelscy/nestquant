import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import mlx.core as mx
import numpy as np

from compact_base import compact, convert_layer, verify
from decoder import projection_matvec_banked
from native_assets import FORMAT, REVISION, bank_storage_bytes, fingerprint, paths, read_bank, restore_compact_alias


def fixture(residual=2.5):
    mx.random.seed(17)
    depth, units, width = 4, 16, 256
    correction = mx.random.normal((depth, 1, width)).astype(mx.float16)
    bank = {}
    for name in ('gate', 'up', 'down'):
        planes = dict(base=mx.random.randint(0, 256, (depth, units * 384), dtype=mx.uint8),
                      var=mx.random.randint(0, 2, (depth, units * 8), dtype=mx.uint8),
                      suh2=mx.ones((depth, width), dtype=mx.float16),
                      svh2=mx.ones((depth, 128), dtype=mx.float16),
                      suh4=mx.ones((depth, width), dtype=mx.float16),
                      svh4=mx.ones((depth, 128), dtype=mx.float16),
                      word=mx.full((depth, units), 200 | (50 << 8), dtype=mx.int32),
                      p4=mx.random.randint(0, 256, (depth, units * int(residual * 256)), dtype=mx.uint8),
                      lrV=mx.array(correction),
                      lrU2=mx.random.normal((depth, 1, 128)).astype(mx.float16),
                      lrU4=mx.random.normal((depth, 1, 128)).astype(mx.float16))
        meta = dict(k=width, n=128, tk=2, tn=8, shard_axis='n', base_K=1.5, res_rule={'K': residual})
        bank[name] = planes, meta
    mx.eval([value for planes, _ in bank.values() for value in planes.values()])
    return bank


class CompactBaseTest(unittest.TestCase):
    def test_roundtrip_preserves_every_byte_and_alias(self):
        original = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / 'source', root / 'target'
            source.mkdir()
            target.mkdir()
            file, sidecar = paths(source, 3)
            mx.save_safetensors(file, {f'{name}.{key}': value for name, (planes, _) in original.items()
                                      for key, value in planes.items()}, metadata=dict(
                format=FORMAT, revision=REVISION, layer='3',
                geometry=json.dumps({name: meta for name, (_, meta) in original.items()})))
            sidecar.write_text(json.dumps(dict(revision=REVISION, state=fingerprint(file),
                                              sha256=hashlib.sha256(file.read_bytes()).hexdigest())))
            evidence = convert_layer(source, target, 3, lambda: None)
            loaded = read_bank(target, 3, 'mlx-nocache')
            compared = verify(original, loaded)
            self.assertEqual(evidence['compared_bytes'], compared)
            saving = 3 * 4 * 16 * 7 + original['gate'][0]['lrV'].nbytes
            self.assertEqual(evidence['original_bytes'] - evidence['stored_bytes'], saving)
            self.assertEqual(bank_storage_bytes(loaded), evidence['stored_bytes'])
            self.assertIs(loaded['gate'][0]['lrV'], loaded['up'][0]['lrV'])
            with self.assertRaises(FileExistsError):
                convert_layer(source, target, 3, lambda: None)

    def test_signs_and_aliases_fail_closed(self):
        bank = fixture()
        bank['gate'][0]['var'] = mx.full(bank['gate'][0]['var'].shape, 2, dtype=mx.uint8)
        with self.assertRaisesRegex(ValueError, 'binary'):
            compact(bank)
        bank = fixture()
        bank['up'][0]['lrV'] = bank['up'][0]['lrV'] + 1
        with self.assertRaisesRegex(ValueError, 'corrections differ'):
            compact(bank)
        arrays, geometry = compact(fixture())
        for aliases in ({}, {'up.lrV': 'down.lrV'}, {'up.lrV': 'gate.lrV', 'gate.lrV': 'up.lrV'}):
            with self.subTest(aliases=aliases), self.assertRaisesRegex(ValueError, 'alias'):
                restore_compact_alias(dict(arrays), geometry, aliases)
        for invalid in (True, 8, '1', 0):
            broken = {name: dict(meta) for name, meta in geometry.items()}
            broken['gate']['var_bits'] = invalid
            with self.subTest(bits=invalid), self.assertRaisesRegex(ValueError, 'sign plane'):
                restore_compact_alias(dict(arrays), broken, {'up.lrV': 'gate.lrV'})

    def test_packed_signs_preserve_gpu_outputs(self):
        for residual in (2.5, 2.8125):
            bank = fixture(residual)
            arrays, geometry = compact(bank)
            route = mx.array((3, 0, 2, 1), dtype=mx.uint32)
            inputs = mx.random.normal((4, 256)).astype(mx.float16).astype(mx.float32)
            for name, (original, meta) in bank.items():
                packed = dict(original, var=arrays[f'{name}.var'])
                for level, mask in ((2, None), (4, None), (2, (3, 0xffffffff, 2, 0xffffffff)),
                                    (2, (0xffffffff,) * 4)):
                    extra = {} if mask is None else dict(refinements=original,
                                                        refined_route=mx.array(mask, dtype=mx.uint32))
                    expected = projection_matvec_banked(original, meta, inputs, route, level, **extra)
                    mx.eval(expected)
                    self.assertTrue(mx.isfinite(expected).all().item())
                    for size in (64, 128, 256, 512):
                        with self.subTest(residual=residual, name=name, level=level, mask=mask, size=size):
                            actual = projection_matvec_banked(packed, geometry[name], inputs, route, level,
                                                             threadgroup_size=size, **extra)
                            self.assertTrue(mx.array_equal(actual, expected).item())
                    invalid = projection_matvec_banked(packed, geometry[name], inputs,
                                                      mx.array((4, 0, 2, 1), dtype=mx.uint32), level, **extra)
                    self.assertTrue(mx.isnan(invalid[0]).all().item())
                    self.assertTrue(mx.array_equal(invalid[1:], expected[1:]).item())
            for bits in (0, 2, True, 1.0):
                with self.subTest(bits=bits), self.assertRaisesRegex(ValueError, 'sign packing'):
                    projection_matvec_banked(bank['gate'][0], dict(bank['gate'][1], var_bits=bits), inputs, route, 2)


if __name__ == '__main__':
    unittest.main()
