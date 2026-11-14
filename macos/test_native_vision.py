import base64
import unittest

import mlx.core as mx
import mlx.nn as nn

import native_vision as nv


class PseudoIdTest(unittest.TestCase):
    def test_round_trip_and_expansion(self):
        key = nv.image_key(b'image bytes')
        self.assertEqual(nv.split_pseudo(nv.pseudo_id(key, 7)), (key, 7))
        self.assertLess(nv.pseudo_id(key, 0), 0)
        ids = nv.expand_prompt([1, 9, 2, 9, 3], 9, [(key, 3), (key + 1, 2)])
        self.assertEqual(len(ids), 8)
        self.assertEqual(ids[0], 1)
        self.assertEqual([nv.split_pseudo(t) for t in ids[1:4]], [(key, 0), (key, 1), (key, 2)])
        self.assertEqual(nv.real_ids(ids, 9), [1, 9, 9, 9, 2, 9, 9, 3])
        # Different images never share a prefix inside the image run.
        self.assertNotEqual(nv.pseudo_id(key, 0), nv.pseudo_id(key + 1, 0))
        with self.assertRaises(nv.ImageError):
            nv.expand_prompt([1, 9, 9], 9, [(key, 1)])
        with self.assertRaises(nv.ImageError):
            nv.expand_prompt([1], 9, [(key, 1)])

    def test_load_image_bytes(self):
        self.assertEqual(nv.load_image_bytes('data:image/png;base64,' + base64.b64encode(b'abc').decode()), b'abc')
        for bad in ('file:///etc/passwd', 'data:image/png,abc', ''):
            with self.assertRaises(nv.ImageError):
                nv.load_image_bytes(bad)


class FeatureEmbeddingTest(unittest.TestCase):
    def test_substitutes_rows_in_order_across_calls(self):
        base = nn.Embedding(16, 4)
        embed = nv.FeatureEmbedding(base, image_token_id=9)
        ids_a, ids_b = mx.array([[1, 9, 9]]), mx.array([[9, 2]])
        plain = embed(ids_a)
        self.assertTrue(mx.array_equal(plain, base(ids_a)).item())
        rows = mx.arange(12, dtype=mx.float32).reshape(3, 4) + 100
        with embed.supplying(rows):
            a, b = embed(ids_a), embed(ids_b)
        self.assertTrue(mx.array_equal(a[0, 0], base(ids_a)[0, 0]).item())
        self.assertTrue(mx.array_equal(a[0, 1:], rows[:2]).item())
        self.assertTrue(mx.array_equal(b[0, 0], rows[2]).item())
        self.assertTrue(mx.array_equal(b[0, 1], base(ids_b)[0, 1]).item())
        # Unused rows mean the prompt and the features disagree.
        with self.assertRaises(RuntimeError):
            with embed.supplying(rows):
                embed(ids_a)
        self.assertTrue(mx.array_equal(embed(ids_a), base(ids_a)).item())


if __name__ == '__main__':
    unittest.main()
