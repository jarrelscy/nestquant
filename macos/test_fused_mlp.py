"""expert_mlp_fused must reproduce projection_fused (gate, up, down) and PagedExperts' SwiGLU bit for bit.
GPU test: run inside the model lock."""
import random
import unittest

import mlx.core as mx

import decoder
from microbench_window import make_layer

COLD = 0xffffffff


def with_low_rank(layer, depth, hot, rank):
    """Add gate/up low-rank corrections like the real checkpoint (up shares gate's V)."""
    gate_base, gate_ref, gate_meta = layer['gate']
    v = (0.05 * mx.random.normal((depth, rank, gate_meta['k']))).astype(mx.float16)
    for name in ('gate', 'up'):
        base, refinements, meta = layer[name]
        base = dict(base, lrV=v, lrU2=(0.05 * mx.random.normal((depth, rank, meta['n']))).astype(mx.float16))
        refinements = dict(refinements, lrU4=(0.05 * mx.random.normal((hot, rank, meta['n']))).astype(mx.float16))
        layer[name] = (base, refinements, meta)
    mx.eval(layer)
    return layer


def reference(layer, tokens, route, refined):
    rows = mx.broadcast_to(tokens[:, None, :], (tokens.shape[0], 8, tokens.shape[1])).reshape(route.size, -1)

    def project(name, value):
        base, refinements, meta = layer[name]
        options = dict(refinements=refinements, refined_route=refined) if refined is not None else {}
        return decoder.projection_fused(base, meta, value, route, **options)

    gate = mx.minimum(project('gate', rows), 10.0)
    up = mx.clip(project('up', rows), -10.0, 10.0)
    return project('down', gate * mx.sigmoid(gate) * up)


def fused(layer, tokens, route, refined):
    planes = {name: (layer[name][0], layer[name][2]) for name in ('gate', 'up', 'down')}
    refinements = {name: layer[name][1] for name in ('gate', 'up', 'down')} if refined is not None else None
    return decoder.expert_mlp_fused(planes, tokens, route, 8, refinements=refinements, refined_route=refined)


class FusedMlpTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mx.random.seed(7)
        random.seed(7)
        cls.depth, cls.hot = 24, 8
        cls.layers = {rank: with_low_rank(make_layer(cls.depth, cls.hot, 1), cls.depth, cls.hot, rank) if rank
                      else make_layer(cls.depth, cls.hot, 1) for rank in (0, 6)}

    def case(self, rank, tokens, mode):
        layer = self.layers[rank]
        pool = random.sample(range(self.depth), min(self.depth, 8 + 3 * tokens))
        ids = [e for _ in range(tokens) for e in random.sample(pool, 8)]
        route = mx.array(ids, dtype=mx.uint32)
        slots = {e: i % self.hot for i, e in enumerate(dict.fromkeys(ids))}
        if mode == 'base':
            refined = None
        elif mode == 'refined':
            refined = mx.array([slots[e] for e in ids], dtype=mx.uint32)
        else:
            refined = mx.array([slots[e] if e % 3 else COLD for e in ids], dtype=mx.uint32)
        x = mx.random.normal((tokens, 4096)).astype(mx.float16).astype(mx.float32)
        want, got = reference(layer, x, route, refined), fused(layer, x, route, refined)
        mx.eval(want, got)
        with self.subTest(rank=rank, tokens=tokens, mode=mode):
            self.assertEqual(got.shape, want.shape)
            self.assertTrue(mx.array_equal(got, want, equal_nan=True).item(),
                            f'max diff {mx.max(mx.abs(got - want)).item()}')

    def test_matches_unfused_path(self):
        for rank in (0, 6):
            for tokens in (1, 3, 8):
                for mode in ('base', 'mixed', 'refined'):
                    self.case(rank, tokens, mode)

    def test_split_precision_launches(self):
        for split, min_rows in ((False, 9), (True, 1)):
            saved = decoder.SPLIT_PRECISION, decoder.SPLIT_MIN_ROWS
            decoder.SPLIT_PRECISION, decoder.SPLIT_MIN_ROWS = split, min_rows
            try:
                for tokens in (1, 8):
                    for mode in ('mixed', 'refined'):
                        self.case(6, tokens, mode)
            finally:
                decoder.SPLIT_PRECISION, decoder.SPLIT_MIN_ROWS = saved

    def test_optional_kernel_variants_are_bit_exact(self):
        """Env-flag variants (NQ_REFINED_MAXR, NQ_REFINED_SPECIALIZED, NQ_FWHT_SIMD, NQ_PAIR_SUM, NQ_HALF_SUM,
        NQ_REFINED_FOLD, NQ_WINDOW_PREFETCH) must reproduce the default kernels bit for bit."""
        factories = ('_window_kernel', '_prep_pair_kernel', '_prep_kernel', '_post_pair_kernel', '_post_kernel')

        def clear():
            for name in factories:
                getattr(decoder, name).cache_clear()

        variants = [dict(REFINED_MAXR=2), dict(REFINED_MAXR=1), dict(REFINED_MAXR=2, SPECIALIZED_MEMBERS=(4, 1)),
                    dict(FWHT_SIMD=True), dict(PAIR_SUM='bfi'), dict(PAIR_SUM='imadk'), dict(HALF_SUM='swar'),
                    dict(REFINED_FOLD='float'), dict(WINDOW_PREFETCH=True)]
        cases = []
        for tokens in (1, 2, 8):
            ids = [e for _ in range(tokens) for e in random.sample(range(12), 8)]   # small pool: groups of 3-8
            route = mx.array(ids, dtype=mx.uint32)
            mixed = mx.array([e % self.hot if e % 3 else COLD for e in ids], dtype=mx.uint32)
            full = mx.array([e % self.hot for e in ids], dtype=mx.uint32)
            x = mx.random.normal((tokens, 4096)).astype(mx.float16).astype(mx.float32)
            cases += [(tokens, mode, x, route, refined) for mode, refined in
                      (('base', None), ('mixed', mixed), ('refined', full))]
        layer = self.layers[6]
        want = [fused(layer, x, route, refined) for *_, x, route, refined in cases]
        mx.eval(want)
        for settings in variants:
            saved = {key: getattr(decoder, key) for key in settings}
            try:
                for key, value in settings.items():
                    setattr(decoder, key, value)
                clear()
                for (tokens, mode, x, route, refined), expected in zip(cases, want):
                    got = fused(layer, x, route, refined)
                    with self.subTest(settings=settings, tokens=tokens, mode=mode):
                        self.assertTrue(mx.array_equal(got, expected, equal_nan=True).item())
                    # The unfused projections share the grouping and rotation kernels.
                    if tokens == 8:
                        reference_out = reference(layer, x, route, refined)
                        with self.subTest(settings=settings, path='unfused', mode=mode):
                            self.assertTrue(mx.array_equal(reference_out, expected, equal_nan=True).item())
            finally:
                for key, value in saved.items():
                    setattr(decoder, key, value)
                clear()

    def test_grouping_disabled(self):
        saved = decoder.GROUP_SHARED_EXPERTS
        decoder.GROUP_SHARED_EXPERTS = False
        try:
            self.case(6, 4, 'mixed')
        finally:
            decoder.GROUP_SHARED_EXPERTS = saved


if __name__ == '__main__':
    unittest.main()
