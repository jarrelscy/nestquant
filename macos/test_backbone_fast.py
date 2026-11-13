import unittest
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn

from mlx_vlm.models.deepseek_v4.hyper_connection import HyperConnection, hc_expand

import backbone_fast as bf


def _hc(dim, seed):
    mx.random.seed(seed)
    config = SimpleNamespace(hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6, rms_norm_eps=1e-5, hidden_size=dim)
    hc = HyperConnection(config)
    hc.fn = mx.random.normal(hc.fn.shape) * 0.02
    hc.base = mx.random.normal(hc.base.shape) * 0.5
    hc.scale = mx.random.uniform(0.5, 1.5, (3,))
    norm = nn.RMSNorm(dim, eps=1e-5)
    norm.weight = (1 + 0.1 * mx.random.normal((dim,))).astype(mx.bfloat16)
    return hc, norm


@unittest.skipUnless(mx.metal.is_available(), 'Metal required')
class HyperConnectionKernelTest(unittest.TestCase):
    dim = 4096

    def test_collapse_matches_reference(self):
        hc, norm = _hc(self.dim, 0)
        x = mx.random.normal((1, 1, 4, self.dim)).astype(mx.bfloat16)
        ref_collapsed, ref_post, ref_comb = hc(x)
        ref = norm(ref_collapsed)
        got, post, comb = bf.hc_finish(bf.hc_partial(x, hc.fn), hc, x, norm)
        self.assertEqual(got.dtype, ref.dtype)
        self.assertLess(mx.abs(post - ref_post).max().item(), 1e-4)
        self.assertLess(mx.abs(comb - ref_comb).max().item(), 1e-4)
        diff = mx.abs(got.astype(mx.float32) - ref.astype(mx.float32))
        scale = mx.abs(ref.astype(mx.float32)).max().item()
        self.assertLess(diff.max().item(), 0.02 * scale)
        # Almost all elements agree to the bfloat16 rounding of the reference.
        self.assertGreater((diff == 0).astype(mx.float32).mean().item(), 0.95)

    def test_expand_matches_reference(self):
        hc, norm = _hc(self.dim, 1)
        nxt, _ = _hc(self.dim, 2)
        x = mx.random.normal((1, 1, 4, self.dim)).astype(mx.bfloat16)
        o = mx.random.normal((1, 1, self.dim)).astype(mx.bfloat16)
        _, post, comb = hc(x)
        ref = hc_expand(o, x, post, comb)
        got, partial = bf.hc_expand_partial(o, x, post, comb, nxt.fn)
        diff = mx.abs(got.astype(mx.float32) - ref.astype(mx.float32))
        self.assertLess(diff.max().item(), 0.02 * mx.abs(ref.astype(mx.float32)).max().item())
        # The reference expand uses an FP32 GEMM whose rounding differs in the last bfloat16 bit for a few elements.
        self.assertGreater((diff == 0).astype(mx.float32).mean().item(), 0.85)
        # The fused partials equal the partials of the expanded streams.
        expected = bf.hc_partial(got, nxt.fn)
        self.assertLess(mx.abs(partial[..., :25] - expected[..., :25]).max().item(), 1e-5 * mx.abs(expected[..., :25]).max().item())

    def test_rows_are_independent(self):
        hc, norm = _hc(self.dim, 3)
        nxt, _ = _hc(self.dim, 4)
        x = mx.random.normal((1, 4, 4, self.dim)).astype(mx.bfloat16)
        o = mx.random.normal((1, 4, self.dim)).astype(mx.bfloat16)
        normed, post, comb = bf.hc_finish(bf.hc_partial(x, hc.fn), hc, x, norm)
        out, partial = bf.hc_expand_partial(o, x, post, comb, nxt.fn)
        for t in range(4):
            xs, os_ = x[:, t:t + 1], o[:, t:t + 1]
            n1, p1, c1 = bf.hc_finish(bf.hc_partial(xs, hc.fn), hc, xs, norm)
            o1, q1 = bf.hc_expand_partial(os_, xs, p1, c1, nxt.fn)
            self.assertTrue(mx.array_equal(n1, normed[:, t:t + 1]).item())
            self.assertTrue(mx.array_equal(c1, comb[:, t:t + 1]).item())
            self.assertTrue(mx.array_equal(o1, out[:, t:t + 1]).item())
            self.assertTrue(mx.array_equal(q1[0, :, :25], partial[t, :, :25]).item())


@unittest.skipUnless(mx.metal.is_available(), 'Metal required')
class BlockLinearTest(unittest.TestCase):
    def test_narrow_block_projection_is_bit_exact(self):
        mx.random.seed(5)
        layer = nn.Linear(128, 8192, bias=False).to_quantized(group_size=64, bits=4)
        for length in (2, 4, 8):
            x = mx.random.normal((1, length, 128)).astype(mx.bfloat16)
            ref = mx.concatenate([layer(x[:, t:t + 1]) for t in range(length)], axis=1)
            self.assertTrue(mx.array_equal(bf._block_linear(layer, x), ref).item())


if __name__ == '__main__':
    unittest.main()
