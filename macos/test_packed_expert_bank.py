import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import mlx.core as mx
import numpy as np

from packed_expert_bank import PackedExpertBank, PackedRefinementBank


class PackedExpertBankTest(unittest.TestCase):
    def test_miss_batches_count_unique_routes_and_attribute_only_required_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'L3.safetensors'
            weights = mx.arange(288 * 4, dtype=mx.int32).reshape(288, 4)
            mx.save_safetensors(path, {'gate.word': weights}, metadata={
                'format': 'nestquant-resident-base-v1', 'revision': 'fixture', 'layer': '3',
                'geometry': json.dumps({'gate': {}}),
            })
            bank = PackedExpertBank(path, 3, 32, 'fixture')
            ticks = iter(value * .01 for value in range(100))
            batches = ((1, 2, 2), (2, 1), (3,), (4, 5, 6), tuple(range(7, 16)),
                       tuple(range(16, 20)), tuple(range(20, 28)))
            try:
                with patch('packed_expert_bank.time.perf_counter', side_effect=lambda: next(ticks)):
                    for experts in batches:
                        route = bank.select(experts)
                        actual = bank.planes['gate'][0]['word'][mx.array(route)]
                        bank.retain(actual)
                        self.assertTrue(mx.array_equal(actual, weights[mx.array(experts)]).item())
                for bucket, misses in ((2, 2), (0, 0), (1, 1), (3, 3), ('ge9', 9), (4, 4), (8, 8)):
                    self.assertEqual(bank.stats[f'miss_batch_{bucket}_calls'], 1)
                    self.assertAlmostEqual(bank.stats[f'miss_batch_{bucket}_read_seconds'], misses * .01)
                self.assertEqual(bank.stats['miss_batch_5_calls'], 0)
                self.assertEqual(bank.stats['miss_batch_6_calls'], 0)
                self.assertEqual(bank.stats['miss_batch_7_calls'], 0)
                self.assertEqual(bank.stats['misses'], 27)
                self.assertAlmostEqual(bank.stats['read_seconds'], .27)
                before = dict(bank.stats)
                with self.assertRaises(ValueError):
                    bank.select((288,))
                self.assertEqual(bank.stats, before)
            finally:
                bank.close()

    def test_paced_preload_checks_before_waiting_and_counts_read_time_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'L3.safetensors'
            weights = mx.arange(288 * 4, dtype=mx.int32).reshape(288, 4)
            mx.save_safetensors(path, {'gate.word': weights, 'up.word': weights}, metadata={
                'format': 'nestquant-resident-base-v1', 'revision': 'fixture', 'layer': '3',
                'geometry': json.dumps({'gate': {}, 'up': {}}),
            })
            bank = PackedExpertBank(path, 3, 288, 'fixture')
            clock = [0.0]
            checks = []
            read = os.preadv

            def delayed_read(*args):
                clock[0] += 0.25
                return read(*args)

            def sleep(seconds):
                self.assertEqual(checks[-1], clock[0])
                self.assertFalse(bank.fully_resident)
                clock[0] += seconds

            try:
                with patch('packed_expert_bank.time.perf_counter', side_effect=lambda: clock[0]), \
                        patch('packed_expert_bank.time.sleep', side_effect=sleep) as waits, \
                        patch('packed_expert_bank.os.preadv', side_effect=delayed_read):
                    bank.preload(lambda: checks.append(clock[0]), bytes_per_second=weights.nbytes)
                self.assertEqual([call.args[0] for call in waits.call_args_list], [0.75, 0.75])
                self.assertEqual(bank.stats['read_seconds'], 0.5)
                self.assertEqual(bank.stats['preload_sleep_seconds'], 1.5)
                self.assertTrue(mx.array_equal(bank.planes['gate'][0]['word'], weights).item())
                self.assertTrue(mx.array_equal(bank.planes['up'][0]['word'], weights).item())
                self.assertTrue(bank.fully_resident)
            finally:
                bank.close()

    def test_paced_preload_does_not_wait_after_a_guard_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'L3.safetensors'
            mx.save_safetensors(path, {'gate.word': mx.zeros((288, 4), dtype=mx.int32)}, metadata={
                'format': 'nestquant-resident-base-v1', 'revision': 'fixture', 'layer': '3',
                'geometry': json.dumps({'gate': {}}),
            })
            bank = PackedExpertBank(path, 3, 288, 'fixture')
            try:
                for invalid in (-1, float('nan'), float('inf')):
                    with self.assertRaises(ValueError):
                        bank.preload(lambda: None, invalid)
                with patch('packed_expert_bank.time.sleep') as sleep:
                    with self.assertRaisesRegex(MemoryError, 'pressure'):
                        bank.preload(Mock(side_effect=MemoryError('pressure')), bytes_per_second=1)
                    sleep.assert_not_called()
                self.assertFalse(bank.fully_resident)
                self.assertEqual(bank.slots, {})
            finally:
                bank.close()

    def test_full_preload_routes_without_further_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'L3.safetensors'
            weights = mx.arange(288 * 4, dtype=mx.int32).reshape(288, 4)
            mx.save_safetensors(path, {'gate.word': weights}, metadata={
                'format': 'nestquant-resident-base-v1', 'revision': 'fixture', 'layer': '3',
                'geometry': json.dumps({'gate': {}}),
            })
            bank = PackedExpertBank(path, 3, 288, 'fixture')
            try:
                with self.assertRaisesRegex(RuntimeError, 'fully loaded'):
                    bank.select_device(mx.array([0], dtype=mx.uint32))
                bank.preload(lambda: None)
                self.assertTrue(bank.fully_resident)
                self.assertEqual(bank.stats['read_bytes'], weights.nbytes)
                self.assertEqual(bank.stats['misses'], 0)
                with patch('packed_expert_bank.os.preadv', side_effect=AssertionError('Unexpected base read')):
                    for selected in ([287, 0, 42, 42], [3, 250, 1]):
                        ids = mx.array(selected, dtype=mx.uint32)
                        route = bank.select_device(ids)
                        with self.assertRaisesRegex(RuntimeError, 'Retain'):
                            bank.select_device(ids)
                        result = bank.planes['gate'][0]['word'][route] + 1
                        bank.retain(result)
                        self.assertTrue(mx.array_equal(result, weights[ids] + 1).item())
                    self.assertEqual(bank.select((287, 0)), (287, 0))
                    bank.retain(bank.planes['gate'][0]['word'].sum())
                self.assertEqual(bank.stats['resident_routes'], 7)
                self.assertEqual(bank.stats['evictions'], 0)
            finally:
                bank.close()
            with self.assertRaisesRegex(RuntimeError, 'fully loaded'):
                bank.select_device(mx.array([0], dtype=mx.uint32))

    def test_failed_preload_does_not_publish_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'L3.safetensors'
            mx.save_safetensors(path, {'gate.word': mx.zeros((288, 4), dtype=mx.int32)}, metadata={
                'format': 'nestquant-resident-base-v1', 'revision': 'fixture', 'layer': '3',
                'geometry': json.dumps({'gate': {}}),
            })
            bank = PackedExpertBank(path, 3, 288, 'fixture')
            try:
                with patch('packed_expert_bank.os.preadv', return_value=0):
                    with self.assertRaisesRegex(EOFError, 'ended early'):
                        bank.preload(lambda: None)
                self.assertFalse(bank.fully_resident)
                self.assertEqual(bank.slots, {})
            finally:
                bank.close()
            bank = PackedExpertBank(path, 3, 32, 'fixture')
            try:
                with self.assertRaisesRegex(RuntimeError, '288 expert rows'):
                    bank.preload(lambda: None)
            finally:
                bank.close()

    def test_guard_failure_does_not_publish_a_partial_preload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'L3.safetensors'
            mx.save_safetensors(path, {'gate.word': mx.zeros((288, 16384), dtype=mx.int32)}, metadata={
                'format': 'nestquant-resident-base-v1', 'revision': 'fixture', 'layer': '3',
                'geometry': json.dumps({'gate': {}}),
            })
            bank = PackedExpertBank(path, 3, 288, 'fixture')
            try:
                with patch('packed_expert_bank.os.preadv', wraps=os.preadv) as reads:
                    with self.assertRaisesRegex(MemoryError, 'guard'):
                        bank.preload(Mock(side_effect=MemoryError('guard')))
                self.assertEqual(reads.call_count, 1)
                self.assertEqual(bank.stats['read_bytes'], 16 * 1024**2)
                self.assertFalse(bank.fully_resident)
                self.assertEqual(bank.slots, {})
            finally:
                bank.close()

    def test_eviction_evaluates_the_previous_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'L3.safetensors'
            weights = mx.arange(288 * 4, dtype=mx.int32).reshape(288, 4)
            mx.save_safetensors(path, {'gate.word': weights}, metadata={
                'format': 'nestquant-resident-base-v1', 'revision': 'fixture', 'layer': '3',
                'geometry': json.dumps({'gate': {}}),
            })
            with self.assertRaisesRegex(MemoryError, 'packed row budget'):
                PackedExpertBank(path, 3, 32, 'fixture', remaining_bytes=511)
            bank = PackedExpertBank(path, 3, 32, 'fixture')
            try:
                self.assertEqual(bank.bytes, 512)
                slots = bank.select(tuple(range(32)))
                with self.assertRaisesRegex(RuntimeError, 'Retain'):
                    bank.select((100,))
                planes, _ = bank.planes['gate']
                previous = planes['word'][slots[0]] + 1
                bank.retain(previous)
                self.assertEqual(bank.select((100,)), (slots[0],))
                self.assertTrue(mx.array_equal(previous, weights[0] + 1).item())
                current = planes['word'][slots[0]] + 1
                bank.retain(current)
                self.assertTrue(mx.array_equal(current, weights[100] + 1).item())
                self.assertEqual(bank.stats['evictions'], 1)
            finally:
                bank.close()

    def test_refinement_overwrite_waits_and_bad_payload_is_not_published(self):
        geometry = dict(k=4, n=8, tk=1, tn=1, res_rule={'K': 2})
        bank = PackedRefinementBank({'gate': ({}, geometry)}, 1)
        first = {'p4': np.full((512,), 17, dtype=np.uint8), 'word': np.ones((1,), dtype=np.int32),
                 'suh4': np.ones((4,), dtype=np.float16), 'svh4': np.ones((8,), dtype=np.float16)}
        self.assertTrue(bank.install(10, {'gate': first}, {10}))
        previous = bank.planes['gate']['p4'][0].sum()
        bank.retain(previous)
        malformed = {**first, 'word': np.ones((2,), dtype=np.int32)}
        with self.assertRaisesRegex(ValueError, 'Missing or malformed'):
            bank.install(20, {'gate': malformed}, {20})
        self.assertEqual(bank.slots, {})
        self.assertEqual(previous.item(), 512 * 17)
        self.assertTrue(bank.install(20, {'gate': first}, {20}))
        self.assertEqual(bank.slots, {20: 0})

    def test_staged_refinement_is_published_only_on_activation(self):
        geometry = dict(k=4, n=8, tk=1, tn=1, res_rule={'K': 2})
        bank = PackedRefinementBank({'gate': ({}, geometry)}, 2)
        payload = {'gate': {'p4': np.full((512,), 5, dtype=np.uint8), 'word': np.ones((1,), dtype=np.int32),
                            'suh4': np.ones((4,), dtype=np.float16), 'svh4': np.ones((8,), dtype=np.float16)}}
        self.assertTrue(bank.install(10, payload, {10}))
        self.assertTrue(mx.array_equal(bank.device_table()[10], mx.array(bank.slots[10], dtype=mx.uint32)).item())
        slot = bank.reserve(20, {10, 20})
        self.assertEqual(slot, 1)
        self.assertIsNone(bank.reserve(20, {10, 20}))
        self.assertTrue(bank.install(20, payload, {10, 20}))    # already staged: no second slot
        self.assertNotIn(20, bank.slots)
        self.assertEqual(int(bank.device_table()[20].item()), PackedRefinementBank.COLD_SLOT)
        bank.write(slot, {'gate': {**payload['gate'], 'p4': np.full((512,), 9, dtype=np.uint8)}})
        bank.activate(20, slot)
        self.assertEqual(bank.slots, {10: 0, 20: 1})
        self.assertEqual(int(bank.device_table()[20].item()), 1)
        self.assertEqual(bank.planes['gate']['p4'][1].sum().item(), 512 * 9)
        # Full bank: the next reservation evicts an unwanted expert, which leaves the table immediately.
        self.assertEqual(bank.reserve(30, {20, 30}), 0)
        self.assertNotIn(10, bank.slots)
        self.assertEqual(int(bank.device_table()[10].item()), PackedRefinementBank.COLD_SLOT)
        bank.release(30, 0)
        self.assertEqual(bank.available, [0])


if __name__ == '__main__':
    unittest.main()
