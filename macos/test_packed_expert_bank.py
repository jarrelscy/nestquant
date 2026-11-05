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


if __name__ == '__main__':
    unittest.main()
