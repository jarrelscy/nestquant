import unittest

import mlx.core as mx

from decoder import projection_matvec_banked


class BankedThreadgroupTest(unittest.TestCase):
    def test_invalid_group_fails_before_reading_planes(self):
        for size in (0, 63, 1024, 64.0, True):
            with self.subTest(size=size), self.assertRaisesRegex(ValueError, 'threads per banked'):
                projection_matvec_banked({}, {}, mx.zeros((1, 128)), (0,), 2, threadgroup_size=size)

    def test_group_sizes_preserve_mixed_outputs_and_invalid_routes(self):
        mx.random.seed(42)
        depth, units, width = 4, 16, 256
        base = dict(base=mx.random.randint(0, 256, (depth, units * 384), dtype=mx.uint8),
                    var=mx.random.randint(0, 2, (depth, units * 8), dtype=mx.uint8),
                    suh2=mx.ones((depth, width), dtype=mx.float16),
                    svh2=mx.ones((depth, 128), dtype=mx.float16),
                    suh4=mx.ones((depth, width), dtype=mx.float16),
                    svh4=mx.ones((depth, 128), dtype=mx.float16),
                    word=mx.full((depth, units), 200 | (50 << 8), dtype=mx.int32),
                    lrV=mx.random.normal((depth, 1, width)).astype(mx.float16),
                    lrU2=mx.random.normal((depth, 1, 128)).astype(mx.float16),
                    lrU4=mx.random.normal((depth, 1, 128)).astype(mx.float16))
        route = mx.array((3, 0, 2, 1), dtype=mx.uint32)
        x = mx.random.normal((4, width)).astype(mx.float16).astype(mx.float32)
        for residual in (2.5, 2.8125):
            meta = dict(k=width, n=128, tk=2, tn=8, shard_axis='n', base_K=1.5, res_rule={'K': residual})
            base['p4'] = mx.random.randint(0, 256, (depth, units * int(residual * 256)), dtype=mx.uint8)
            for level, refined in ((2, None), (4, None), (2, (3, 0xffffffff, 2, 0xffffffff)),
                                   (2, (0xffffffff,) * 4)):
                extra = {} if refined is None else dict(refinements=base, refined_route=mx.array(refined, dtype=mx.uint32))
                expected = projection_matvec_banked(base, meta, x, route, level, **extra)
                mx.eval(expected)
                self.assertTrue(mx.isfinite(expected).all().item())
                for size in (64, 128, 256, 512):
                    with self.subTest(residual=residual, level=level, refined=refined, size=size):
                        actual = projection_matvec_banked(base, meta, x, route, level,
                                                         threadgroup_size=size, **extra)
                        self.assertTrue(mx.array_equal(actual, expected).item())
                        invalid = projection_matvec_banked(base, meta, x, mx.array((4, 0, 2, 1), dtype=mx.uint32),
                                                          level, threadgroup_size=size, **extra)
                        self.assertTrue(mx.isnan(invalid[0]).all().item())
                        self.assertTrue(mx.isfinite(invalid[1:]).all().item())


if __name__ == '__main__':
    unittest.main()
