import unittest
import unittest.mock
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models.cache import KVCache, PoolingCache
from mlx_vlm.models.glm5_next import language
from mlx_vlm.speculative.cache_state import start_speculative_cache

import backbone_fast as bf
import sparse_kernels as sk


def _indexer(seed, hidden=512, q_rank=256):
    mx.random.seed(seed)
    config = SimpleNamespace(index_n_heads=32, index_head_dim=128, index_topk=2048, index_kpool=4,
                             index_kpool_always_select_tail=True, q_lora_rank=q_rank, hidden_size=hidden)
    indexer = language.Glm5NextIndexer(config, 0)
    indexer.index_kpool_compress_ape = mx.random.normal(indexer.index_kpool_compress_ape.shape) * 0.1
    indexer.index_kpool_compress_gate = mx.random.normal(indexer.index_kpool_compress_gate.shape) * 0.05
    indexer.update(nn.utils.tree_map(lambda v: v.astype(mx.bfloat16) if v.ndim == 2 and v.shape[-1] in (hidden, q_rank)
                                     else v, indexer.parameters()))
    return indexer.eval(), config


def _prefill(indexer, config, length, seed):
    mx.random.seed(seed)
    caches = (KVCache(), PoolingCache(4))
    x = mx.random.normal((1, length, config.hidden_size)).astype(mx.bfloat16)
    q = mx.random.normal((1, length, config.q_lora_rank)).astype(mx.bfloat16)
    for start in range(0, length, 512):
        out = indexer(x[:, start:start + 512], q[:, start:start + 512], None, caches[0], caches[1], start)
        mx.eval(out, caches[0].state, caches[1].state)
    return caches


def _trim(reference, pools):
    region = 4 * min(512, pools)
    return mx.concatenate([reference[..., :region], reference[..., 2048:]], axis=-1)


@unittest.skipUnless(mx.metal.is_available(), 'Metal required')
class IndexSelectTest(unittest.TestCase):
    def check(self, context, block, seed):
        indexer, config = _indexer(seed)
        caches = _prefill(indexer, config, context, seed + 1)
        mx.random.seed(seed + 2)
        x = mx.random.normal((1, block, config.hidden_size)).astype(mx.bfloat16)
        q = mx.random.normal((1, block, config.q_lora_rank)).astype(mx.bfloat16)
        transaction = start_speculative_cache(list(caches), block)
        ref = indexer(x, q, None, caches[0], caches[1], context)
        mx.eval(ref)
        transaction.abort()
        pools = (context + block) // 4
        with unittest.mock.patch.object(bf, 'TRIM_TOPK', True):
            transaction = start_speculative_cache(list(caches), block)
            got = bf.fast_indexer(indexer, x, q, caches[0], caches[1], context)
            mx.eval(got)
            transaction.abort()
        expected = _trim(ref, pools)
        self.assertEqual(got.shape, expected.shape)
        self.assertTrue(mx.array_equal(got, expected).item(), f'context {context} block {block}')
        with unittest.mock.patch.object(bf, 'TRIM_TOPK', False):
            transaction = start_speculative_cache(list(caches), block)
            full = bf.fast_indexer(indexer, x, q, caches[0], caches[1], context)
            mx.eval(full)
            transaction.abort()
        self.assertTrue(mx.array_equal(full, ref).item(), f'untrimmed context {context} block {block}')

    def test_select_all_short_context(self):
        for context, block in ((997, 1), (1000, 4), (1203, 8), (2040, 8)):
            self.check(context, block, 10 + block)

    def test_top_k_long_context(self):
        for context, block in ((2045, 4), (2051, 1), (4097, 8), (9001, 4), (9003, 1)):
            self.check(context, block, 20 + block)

    def test_scores_match_mlx_bitwise(self):
        # The kernel's head-weighted score is the reference's float32 arithmetic in the same order: equal
        # scores give an identical selection even for pools whose scores differ in the last bit.
        mx.random.seed(5)
        rows, heads, pools = 3, 32, 3000
        raw = mx.random.normal((rows, heads, pools))
        weights = mx.random.normal((rows, heads))
        scale = mx.array([128 ** -0.5], dtype=mx.float32)
        ref = (mx.maximum(raw, 0) * (weights * 128 ** -0.5)[..., None]).sum(axis=1)
        valid = mx.ones((4 * pools + 8,), dtype=mx.bool_)
        got, keys = sk.index_select(raw, weights, scale, valid, pools, pools, 512, rows, return_keys=True)
        bits = (ref + 0.0).view(mx.uint32)
        expected_keys = mx.where(bits >= 2 ** 31, ~bits, bits | (2 ** 31))
        self.assertTrue(mx.array_equal(keys, expected_keys).item())
        for row in range(rows):
            order = mx.argsort(-ref[row])[:512]
            expected = mx.sort(order)
            self.assertTrue(mx.array_equal(got[row, :2048:4] // 4, expected).item())


if __name__ == '__main__':
    unittest.main()
