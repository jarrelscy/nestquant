import unittest
from unittest.mock import Mock

import mlx.core as mx
import numpy as np

from native_refinements import COLD_SLOT, NativeHotRefinements
from packed_expert_bank import PackedRefinementBank


def empty_bank(rows):
    bank = object.__new__(PackedRefinementBank)
    bank.slots = {}
    bank.available = list(reversed(range(rows)))
    bank.host = {}
    bank.previous_output = None
    bank.table = np.full(288, COLD_SLOT, dtype=np.uint32)
    bank._device_table = None
    return bank


class DeviceSlotTableTest(unittest.TestCase):
    def test_table_tracks_installs_and_evictions(self):
        bank = empty_bank(2)
        self.assertTrue(bank.install(5, {}, {5}))
        self.assertTrue(bank.install(9, {}, {5, 9}))
        first = bank.device_table()
        self.assertIs(bank.device_table(), first)
        self.assertEqual(first[mx.array([5, 9, 7], dtype=mx.uint32)].tolist(), [0, 1, COLD_SLOT])
        self.assertFalse(bank.install(7, {}, {5, 9}))
        self.assertTrue(bank.install(7, {}, {9, 7}))
        table = bank.device_table()
        self.assertIsNot(table, first)
        self.assertEqual(table[mx.array([5, 9, 7], dtype=mx.uint32)].tolist(), [COLD_SLOT, 1, 0])


class DeviceCommitTest(unittest.TestCase):
    def test_commit_reads_device_routes_once_per_token(self):
        hot = object.__new__(NativeHotRefinements)
        hot.phase = 'decode'
        hot.chunk_index = 0
        hot.position = 3
        hot.seconds = 0.0
        hot.ready = np.zeros((42, 288), dtype=bool)
        hot.wanted = np.zeros((42, 288), dtype=bool)
        hot.metrics = Mock()
        hot.predictor = Mock(block=None, policies=[Mock(state=np.zeros(1))])
        hot.banks = {}
        hot.traces = {}
        for layer in range(3, 45):
            bank = empty_bank(4)
            bank.install(layer % 288, {}, {layer % 288})
            hot.banks[layer] = bank
            ids = mx.array([layer % 288, 100, 101, 102, 103, 104, 105, 106], dtype=mx.uint32)
            weights = mx.full((1, 8), 0.125)
            norms = mx.array([1.0])
            _, slots = hot.route_device(layer, ids, weights, norms)
            self.assertEqual(slots.tolist()[0], 0)
        hot.commit([42])
        self.assertTrue(all(hot.ready[layer - 3, layer % 288] for layer in range(3, 45)))
        self.assertEqual(int(hot.ready.sum()), 42)
        (rows,), _ = hot.predictor.commit.call_args
        position, token, selected, weights, norms = rows[0]
        self.assertEqual((position, token, selected.shape, weights.shape, norms.shape), (3, 42, (42, 8), (42, 8), (42,)))
        self.assertEqual(int(selected[0, 0]), 3)
        self.assertEqual(hot.position, 4)

    def test_device_routing_is_limited_to_decode(self):
        hot = object.__new__(NativeHotRefinements)
        hot.phase = 'prefill'
        with self.assertRaisesRegex(RuntimeError, 'limited to decode'):
            hot.route_device(3, mx.array([1], dtype=mx.uint32), None, None)


class WarmPoolTest(unittest.TestCase):
    def test_warm_installs_each_initial_pool(self):
        hot = object.__new__(NativeHotRefinements)
        hot.predictor = Mock(layers=[3, 4], defaults=[np.array([5, 9]), np.array([1, 2, 3])])
        hot.banks = {3: empty_bank(2), 4: empty_bank(3)}
        hot.reader = Mock(read=Mock(side_effect=lambda layer, expert: ({}, 10, 0.0)))
        checks = Mock()
        report = hot.warm(readers=1, check=checks)
        self.assertEqual(report['installed'], 5)
        self.assertEqual(report['read_bytes'], 50)
        self.assertEqual(set(hot.banks[3].slots), {5, 9})
        self.assertEqual(set(hot.banks[4].slots), {1, 2, 3})
        self.assertEqual(checks.call_count, 5)
        self.assertEqual(hot.warm(readers=1)['installed'], 0)

    def test_warm_rejects_a_pool_larger_than_its_bank(self):
        hot = object.__new__(NativeHotRefinements)
        hot.predictor = Mock(layers=[3], defaults=[np.array([1, 2, 3])])
        hot.banks = {3: empty_bank(2)}
        hot.reader = Mock(read=Mock(return_value=({}, 1, 0.0)))
        with self.assertRaisesRegex(RuntimeError, 'does not fit'):
            hot.warm(readers=1)


if __name__ == '__main__':
    unittest.main()
