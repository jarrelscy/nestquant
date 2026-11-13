import unittest

import mlx.core as mx

from decoder import projection_matvec_banked
from microbench_window import GEOMETRY, make_layer

COLD = 0xffffffff


def relative_rms(value, reference):
    return float(mx.sqrt(mx.mean(mx.square(value - reference)) / mx.mean(mx.square(reference))).item())


class WindowDecoderTest(unittest.TestCase):
    """The window loop must match the inline loop up to float32 summation order."""

    @classmethod
    def setUpClass(cls):
        mx.random.seed(3)
        cls.layers = {var_bits: make_layer(12, 4, var_bits) for var_bits in (1, 8)}

    def compare(self, var_bits, level, refined_route=None, rows=8):
        route = mx.array([11, 0, 5, 7, 3, 9, 1, 6][:rows], dtype=mx.uint32)
        for name in GEOMETRY:
            base, refinements, meta = self.layers[var_bits][name]
            planes = dict(base)
            if level == 4:
                planes.update(p4=mx.tile(refinements['p4'], (3, 1)), word=mx.tile(refinements['word'], (3, 1)),
                              suh4=mx.tile(refinements['suh4'], (3, 1)), svh4=mx.tile(refinements['svh4'], (3, 1)))
            x = mx.random.normal((rows, meta['k']))
            results = {}
            for mode in ('inline', 'window'):
                results[mode] = projection_matvec_banked(
                    planes, meta, x, route, level, loop_mode=mode,
                    refinements=refinements if refined_route is not None else None, refined_route=refined_route)
            mx.eval(results)
            with self.subTest(name=name, var_bits=var_bits, level=level, mixed=refined_route is not None):
                self.assertLess(relative_rms(results['window'], results['inline']), 1e-4)

    def test_base_level_matches_inline(self):
        for var_bits in (1, 8):
            self.compare(var_bits, 2)

    def test_refined_level_matches_inline(self):
        for var_bits in (1, 8):
            self.compare(var_bits, 4)

    def test_mixed_rows_match_inline(self):
        refined = mx.array([0, COLD, 3, COLD, 1, 2, COLD, 0], dtype=mx.uint32)
        for var_bits in (1, 8):
            self.compare(var_bits, 2, refined)

    def test_single_row(self):
        self.compare(1, 2, mx.array([2], dtype=mx.uint32), rows=1)

    def test_window_rejects_inline_only_options(self):
        base, _, meta = self.layers[1]['gate']
        x = mx.zeros((1, meta['k']))
        route = mx.array([0], dtype=mx.uint32)
        for options in (dict(sum_lookup=True), dict(state_load='u32'), dict(threadgroup_size=128)):
            with self.subTest(options=options), self.assertRaises(ValueError):
                projection_matvec_banked(base, meta, x, route, 2, loop_mode='window', **options)


if __name__ == '__main__':
    unittest.main()


class FusedProjectionTest(unittest.TestCase):
    """The three-kernel path must match the window path, including low-rank terms and invalid slots."""

    @classmethod
    def setUpClass(cls):
        mx.random.seed(5)
        cls.layer = make_layer(12, 4, 1)
        for name, (base, refinements, meta) in cls.layer.items():
            rank = 2
            base['lrV'] = (0.01 * mx.random.normal((12, rank, meta['k']))).astype(mx.float16)
            base['lrU2'] = (0.01 * mx.random.normal((12, rank, meta['n']))).astype(mx.float16)
            refinements['lrU4'] = (0.01 * mx.random.normal((4, rank, meta['n']))).astype(mx.float16)
        mx.eval(cls.layer)

    def test_fused_matches_window(self):
        from decoder import projection_fused
        route = mx.array([11, 0, 5, 7, 3, 9, 1, 6], dtype=mx.uint32)
        for refined in (None, mx.array([0, COLD, 3, COLD, 1, 2, COLD, 7], dtype=mx.uint32)):
            for name, (base, refinements, meta) in self.layer.items():
                x = mx.random.normal((8, meta['k']))
                options = dict(refinements=refinements, refined_route=refined) if refined is not None else {}
                expected = projection_matvec_banked(base, meta, x, route, 2, loop_mode='window', **options)
                actual = projection_fused(base, meta, x, route, **options)
                mx.eval(expected, actual)
                with self.subTest(name=name, mixed=refined is not None):
                    self.assertEqual(mx.isnan(expected).all(axis=1).tolist(), mx.isnan(actual).all(axis=1).tolist())
                    keep = mx.array([i for i, bad in enumerate(mx.isnan(expected).any(axis=1).tolist()) if not bad])
                    self.assertLess(relative_rms(actual[keep], expected[keep]), 1e-5)


class GroupedRowsTest(unittest.TestCase):
    """Rows sharing an expert decode once; results must match the ungrouped kernel."""

    def test_grouped_matches_ungrouped(self):
        import decoder
        from decoder import projection_fused
        mx.random.seed(9)
        layer = make_layer(12, 4, 1)
        # 32 rows (4 tokens x 8 experts) with heavy expert reuse, including runs longer than MAXR.
        route = mx.array([3, 3, 3, 3, 5, 7, 9, 11] * 2 + [3, 5, 5, 0, 1, 2, 4, 6] + [3, 3, 8, 8, 10, 1, 2, 5],
                         dtype=mx.uint32)
        refined = mx.array([(i * 7) % 4 if i % 3 else COLD for i in range(32)], dtype=mx.uint32)
        # Same expert must map to the same slot within a call, as the runtime guarantees.
        slot_of = {}
        values = []
        for expert, slot in zip(route.tolist(), refined.tolist()):
            values.append(slot_of.setdefault(expert, slot))
        refined = mx.array(values, dtype=mx.uint32)
        for name, (base, refinements, meta) in layer.items():
            x = mx.random.normal((32, meta['k']))
            for options in ({}, dict(refinements=refinements, refined_route=refined)):
                decoder.GROUP_SHARED_EXPERTS = False
                expected = projection_fused(base, meta, x, route, **options)
                decoder.GROUP_SHARED_EXPERTS = True
                actual = projection_fused(base, meta, x, route, **options)
                mx.eval(expected, actual)
                with self.subTest(name=name, mixed=bool(options)):
                    self.assertLess(relative_rms(actual, expected), 1e-6)


class WindowVariantsTest(unittest.TestCase):
    """Member-count specializations, group sizes and decode variants against the default kernel."""

    @classmethod
    def setUpClass(cls):
        mx.random.seed(11)
        cls.layer = make_layer(12, 4, 1)
        # 64 rows: runs of one expert from 1 to 9 rows, refined and base-only, plus an invalid slot.
        experts = [0] * 9 + [1] * 7 + [2] * 6 + [3] * 5 + [4] * 4 + [5] * 3 + [6] * 2 + [7] + [8, 9, 10, 11] * 6 + [2, 3, 5]
        cls.route = mx.array(experts[:64], dtype=mx.uint32)
        slots = {0: 0, 2: 1, 4: 2, 6: 3, 8: 0, 10: 9}
        cls.refined = mx.array([slots.get(e, COLD) for e in experts[:64]], dtype=mx.uint32)

    def run_all(self, **settings):
        import decoder
        saved = {name: getattr(decoder, name) for name in settings}
        try:
            for name, value in settings.items():
                setattr(decoder, name, value)
            outputs = []
            for name, (base, refinements, meta) in self.layer.items():
                x = mx.random.normal((64, meta['k']), key=mx.random.key(len(name)))
                for options in ({}, dict(refinements=refinements, refined_route=self.refined)):
                    outputs.append(decoder.projection_fused(base, meta, x, self.route, **options))
            mx.eval(outputs)
            return outputs
        finally:
            for name, value in saved.items():
                setattr(decoder, name, value)

    def test_group_sizes_are_bit_exact(self):
        for positions in (8, 16):
            expected = self.run_all(GROUP_SHARED_EXPERTS=False, LANE_POSITIONS=positions)
            for maxr in (2, 3, 4, 5, 8):
                actuals = self.run_all(GROUP_MAXR_OVERRIDE=maxr, LANE_POSITIONS=positions)
                for index, (actual, reference) in enumerate(zip(actuals, expected)):
                    with self.subTest(maxr=maxr, positions=positions, mixed=bool(index % 2)):
                        if positions == 16 and index % 2:
                            # Refined rows with partners decode eight positions per lane: equal up to
                            # float32 summation order.
                            self.assertTrue(mx.array_equal(mx.isnan(actual), mx.isnan(reference)).item())
                            keep = mx.array([i for i, bad in enumerate(mx.isnan(reference).any(axis=1).tolist())
                                             if not bad])
                            self.assertLess(relative_rms(actual[keep], reference[keep]), 1e-6)
                        else:
                            self.assertTrue(mx.array_equal(actual, reference, equal_nan=True).item())

    def test_specializations_are_bit_exact(self):
        expected = self.run_all(SPECIALIZED_MEMBERS=(1, 1), GROUP_MAXR_OVERRIDE=8)
        for spec in ((4, 1), (8, 8), (2, 4)):
            for actual, reference in zip(self.run_all(SPECIALIZED_MEMBERS=spec, GROUP_MAXR_OVERRIDE=8), expected):
                with self.subTest(spec=spec):
                    self.assertTrue(mx.array_equal(actual, reference, equal_nan=True).item())

    def test_lane_positions_match_up_to_summation_order(self):
        results = {positions: self.run_all(LANE_POSITIONS=positions) for positions in (8, 16)}
        for actual, reference in zip(results[16], results[8]):
            self.assertTrue(mx.array_equal(mx.isnan(actual), mx.isnan(reference)).item())
            keep = mx.array([i for i, bad in enumerate(mx.isnan(reference).any(axis=1).tolist()) if not bad])
            self.assertLess(relative_rms(actual[keep], reference[keep]), 1e-6)
