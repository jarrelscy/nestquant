"""kvshare: LMCache key derivation (spec worked example) and value decoding helpers."""

import unittest

import numpy as np

import kvshare as kv


class KeysTest(unittest.TestCase):
    def test_worked_example(self):
        toks = [(i * 7919) % 154880 for i in range(14436)]
        hashes = kv.chunk_hashes(toks)
        self.assertEqual(len(hashes), 2)
        self.assertEqual(kv.NONE_HASH.hex(), '7e3eff9ef7613a50340eb12677ee03b83ac759fcba9bb32705f07248d56d473b')
        self.assertEqual(hashes[0].hex(), 'c5b37bff05a28e53cbd0a89be873ebf34ac6bd60f396aceab1f0334fc0e02062')
        self.assertEqual(hashes[1].hex(), '3b0603c123b578fa44519711bb5fef3ba6ebc7fc4145eb7b8d891cce1cb9eee6')
        self.assertEqual(kv.object_key(hashes[0], 1, 1),
                         f'{kv.MODEL}@02010201@1@c5b37bff05a28e53cbd0a89be873ebf34ac6bd60f396aceab1f0334fc0e02062')
        self.assertEqual(kv.chunk_hashes(toks[:7167]), [])


class DecodeTest(unittest.TestCase):
    def test_e4m3_and_hadamard(self):
        self.assertEqual(kv.E4M3[0x7E], 448.0)
        self.assertEqual(kv.E4M3[0x38], 1.0)
        self.assertTrue(np.isnan(kv.E4M3[0x7F]))
        self.assertEqual(kv.E4M3[0x01], 2.0 ** -9)
        self.assertTrue(np.allclose(kv.HADAMARD @ kv.HADAMARD, np.eye(128), atol=1e-5))

    def test_mla_interleave_and_scales(self):
        data = [bytearray(kv.G0_BYTES) for _ in range(2)]
        for r in range(2):
            rows = np.zeros((kv.MLA_ROWS, kv.MLA_RECORD), dtype=np.uint8)
            rows[:, :512] = 0x38                                   # 1.0
            rows[:, 512:528] = np.array([r + 1] * 4, dtype='<f4').view(np.uint8)
            data[r][kv.MLA_OFFSET:kv.MLA_OFFSET + kv.MLA_LAYER_BYTES] = rows.tobytes()
        latent = kv.decode_mla((bytes(data[0]), bytes(data[1])), 0)
        self.assertEqual(latent.shape, (kv.CHUNK, 512))
        self.assertTrue((latent[0::2] == 1.0).all() and (latent[1::2] == 2.0).all())


class ReconnectTest(unittest.TestCase):
    def test_dropped_connection_is_replaced(self):
        class Dead:
            closed = False
            def pipeline(self, commands): raise ConnectionError('Kvrocks closed the connection')
            def close(self): self.closed = True
        class Live:
            def pipeline(self, commands): return [1] * len(commands)
            def close(self): pass
        shared = kv.SharedKV('h')
        dead = Dead()
        shared._pool = [dead]
        made = []
        shared._client = lambda: shared._pool.pop() if shared._pool else made.append(Live()) or made[-1]
        self.assertEqual(shared._call(lambda c: c.pipeline([1, 2])), [1, 1])
        self.assertTrue(dead.closed)
        self.assertEqual(len(made), 1)


if __name__ == '__main__':
    unittest.main()
