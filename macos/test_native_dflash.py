import unittest
from types import SimpleNamespace

import mlx.core as mx

import native_dflash as nd


class FakeDrafter:
    def __init__(self):
        self.config = SimpleNamespace(block_size=8)
        self.calls = []

    def draft_block(self, last_bonus, hidden, cache, block_size, sampler, token_dtype=mx.int32):
        self.calls.append(block_size)
        return mx.arange(block_size - 1, dtype=token_dtype)[None]


class PositionAcceptanceTest(unittest.TestCase):
    def test_counts_only_rounds_that_drafted_the_position(self):
        self.assertEqual(nd.position_acceptance([3, 1, 0, 2], [3, 3, 3, 2]), [0.75, 0.5, 0.333])


if __name__ == '__main__':
    unittest.main()
