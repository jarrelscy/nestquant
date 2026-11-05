import unittest
from unittest.mock import patch

import mlx.core as mx
import numpy as np

from decoder import HEADER, PATTERNS, _matvec_kernel, _sum_table, projection_matvec_banked
from test_compact_base import compact, fixture


class DecoderVariantsTest(unittest.TestCase):
    def test_shuffle_rejects_more_words_than_lanes(self):
        with patch.dict(PATTERNS, {5.0: (5, 0)}), self.assertRaisesRegex(ValueError, 'cooperative ring layout'):
            _matvec_kernel(5.0, 2.5, state_load='shuffle')

    def test_wide_reads_match_circular_bit_reference(self):
        kernel = mx.fast.metal_kernel(
            name='test_nq_packed_states', input_names=['buffer'], output_names=['ordinary', 'wide', 'shuffled'], header=HEADER,
            source='''uint i = thread_position_in_grid.x;
            if (i >= RINGS * 256) return;
            ordinary[i] = nq_state(buffer, i / 256, i % 256, KA, MASK);
            wide[i] = nq_state_wide(buffer, i / 256, i % 256, KA, MASK);
            uint word = nq_ring_word(buffer, i / 256, thread_index_in_simdgroup, KA, MASK);
            shuffled[i] = nq_state_shuffled(word, i % 256, KA, MASK);''')
        rng = np.random.default_rng(42)
        for density, (ka, mask) in PATTERNS.items():
            ring_bytes = int(density * 32)
            for count in (1, 2, 3, 17):
                for kind in ('zero', 'ones', 'random'):
                    raw = (np.zeros((count, ring_bytes), dtype=np.uint8) if kind == 'zero' else
                           np.full((count, ring_bytes), 255, dtype=np.uint8) if kind == 'ones' else
                           rng.integers(0, 256, (count, ring_bytes), dtype=np.uint8))
                    expected = []
                    for ring in raw:
                        bits = ring_bytes * 8
                        value = int.from_bytes(ring.tobytes(), 'little')
                        doubled = value | (value << bits)
                        for position in range(256):
                            bit = ka * position + sum((mask >> (index % 16)) & 1 for index in range(position))
                            expected.append((doubled >> bit) & 65535)
                    for offset in range(4):
                        buffer = mx.concatenate([mx.zeros((offset,), dtype=mx.uint8), mx.array(raw.reshape(-1))])
                        mx.eval(buffer)
                        original, actual, shuffled = kernel(inputs=[buffer[offset:]],
                                                  template=[('RINGS', count), ('KA', ka), ('MASK', mask)],
                                                  grid=(count * 256, 1, 1), threadgroup=(256, 1, 1),
                                                  output_shapes=[(count * 256,)] * 3, output_dtypes=[mx.uint32] * 3)
                        with self.subTest(density=density, count=count, kind=kind, offset=offset):
                            self.assertTrue(mx.array_equal(original, mx.array(expected, dtype=mx.uint32)).item())
                            self.assertTrue(mx.array_equal(actual, original).item())
                            self.assertTrue(mx.array_equal(shuffled, original).item())

    def test_lookup_covers_every_state_exactly(self):
        expected = []
        for state in range(65536):
            value = (state * 0x83DCD12D) & 0xffffffff
            expected.append(sum(value.to_bytes(4, 'little')))
        self.assertTrue(mx.array_equal(_sum_table(), mx.array(expected, dtype=mx.uint16)).item())

    def test_variants_preserve_all_projection_modes(self):
        for residual in (2.5, 2.8125):
            bank = fixture(residual)
            arrays, geometry = compact(bank)
            route = mx.array((3, 0, 2, 1), dtype=mx.uint32)
            inputs = mx.random.normal((4, 256)).astype(mx.float16).astype(mx.float32)
            for name, (planes, meta) in bank.items():
                for level, mask in ((2, None), (4, None), (2, (3, 0xffffffff, 2, 0xffffffff)),
                                    (2, (0xffffffff,) * 4)):
                    extra = {} if mask is None else dict(refinements=planes, refined_route=mx.array(mask, dtype=mx.uint32))
                    expected = projection_matvec_banked(planes, meta, inputs, route, level, **extra)
                    for compacted in (False, True):
                        selected = dict(planes, var=arrays[f'{name}.var']) if compacted else planes
                        metadata = geometry[name] if compacted else meta
                        for state_load, lookup, group in [(load, table, group) for load, table in (
                                ('u32', False), ('bytes', True), ('u32', True), ('shuffle', False), ('shuffle', True))
                                for group in (64, 128, 256, 512)]:
                            with self.subTest(residual=residual, name=name, level=level, mask=mask,
                                              compacted=compacted, state_load=state_load, lookup=lookup, group=group):
                                actual = projection_matvec_banked(selected, metadata, inputs, route, level,
                                                                 state_load=state_load, sum_lookup=lookup,
                                                                 threadgroup_size=group, **extra)
                                self.assertTrue(mx.array_equal(actual, expected).item())
                                invalid = projection_matvec_banked(selected, metadata, inputs,
                                                                  mx.array((4, 0, 2, 1), dtype=mx.uint32), level,
                                                                  state_load=state_load, sum_lookup=lookup,
                                                                  threadgroup_size=group, **extra)
                                self.assertTrue(mx.isnan(invalid[0]).all().item())
                                self.assertTrue(mx.array_equal(invalid[1:], expected[1:]).item())

    def test_invalid_variants_fail_before_reading_planes(self):
        for options in (dict(state_load='u64'), dict(sum_lookup=1), dict(sum_lookup='yes')):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, 'decoder variant'):
                projection_matvec_banked({}, {}, mx.zeros((1, 128)), (0,), 2, **options)


if __name__ == '__main__':
    unittest.main()
