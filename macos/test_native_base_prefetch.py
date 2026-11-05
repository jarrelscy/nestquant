import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import mlx.core as mx
import numpy as np

from native_base_prefetch import NativeBasePrefetch
from packed_expert_bank import PackedExpertBank


class NativeBasePrefetchTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.weights = mx.arange(288 * 4, dtype=mx.int32).reshape(288, 4)
        self.banks = []
        for layer in range(3, 45):
            path = Path(self.directory.name) / f'L{layer}.safetensors'
            mx.save_safetensors(path, {'gate.word': self.weights}, metadata={
                'format': 'nestquant-resident-base-v1', 'revision': 'fixture', 'layer': str(layer),
                'geometry': json.dumps({'gate': {}}),
            })
            self.banks.append(PackedExpertBank(path, layer, 32, 'fixture'))
        self.prefetch = NativeBasePrefetch(self.banks)
        self.priority = np.zeros((42, 288))
        self.priority[:, 100] = 2
        self.priority[:, 101] = 1

    def tearDown(self):
        self.prefetch.close()
        for bank in self.banks:
            bank.close()
        self.directory.cleanup()

    def ready(self):
        for future, _ in list(self.prefetch.worker.pending.values()):
            future.result(timeout=5)

    def test_prediction_does_not_publish_and_required_install_preserves_pending_output(self):
        bank = self.banks[0]
        rows = bank.select(tuple(range(32)))
        old = bank.planes['gate'][0]['word'][rows[0]] + 1
        bank.retain(old)
        self.prefetch.begin(self.priority)
        self.ready()
        self.assertNotIn(100, bank.slots)
        with patch.object(bank, '_fill', side_effect=AssertionError('Unexpected synchronous read')):
            row = bank.select((100, 100))[0]
        self.assertEqual(row, rows[0])
        self.assertTrue(mx.array_equal(old, self.weights[0] + 1).item())
        result = bank.planes['gate'][0]['word'][row] + 1
        bank.retain(result)
        self.assertTrue(mx.array_equal(result, self.weights[100] + 1).item())
        self.assertEqual(bank.stats['prefetch_hits'], 1)
        self.assertEqual(bank.stats['misses'], 33)
        self.assertLessEqual(self.prefetch.worker.peak_reserved_bytes, self.prefetch.worker.budget_bytes)

    def test_prediction_miss_reads_required_expert_and_discards_unused_rows(self):
        self.prefetch.begin(self.priority)
        self.ready()
        bank = self.banks[0]
        row = bank.select((200,))[0]
        result = bank.planes['gate'][0]['word'][row]
        bank.retain(result)
        self.assertTrue(mx.array_equal(result, self.weights[200]).item())
        self.assertEqual(set(bank.slots), {200})
        self.assertEqual(bank.stats['prefetch_hits'], 0)
        self.assertEqual(bank.stats['prefetch_discarded'], 2)
        self.prefetch.drain()
        self.assertEqual(self.prefetch.worker.reserved_bytes, 0)

    def test_speculative_io_error_falls_back_and_unused_errors_are_counted(self):
        bank = self.banks[0]
        with patch.object(bank, 'read_payload', side_effect=OSError('SSD failed')):
            self.prefetch.begin(self.priority)
            future = self.prefetch.worker.pending[3, 100][0]
            with self.assertRaisesRegex(OSError, 'SSD failed'):
                future.result(timeout=5)
            second = self.prefetch.worker.pending[3, 101][0]
            with self.assertRaisesRegex(OSError, 'SSD failed'):
                second.result(timeout=5)
            row = bank.select((100,))[0]
            actual = bank.planes['gate'][0]['word'][row]
            bank.retain(actual)
            self.assertTrue(mx.array_equal(actual, self.weights[100]).item())
        self.assertEqual(bank.stats['prefetch_errors'], 2)
        self.assertEqual(bank.stats['prefetch_hits'], 0)
        self.assertEqual(bank.stats['read_bytes'], 16)
        self.prefetch.drain()
        self.assertEqual(self.prefetch.worker.reserved_bytes, 0)

    def test_late_unwanted_read_is_retired_without_blocking_or_installation(self):
        entered, release = threading.Event(), threading.Event()
        bank = self.banks[0]
        original = bank.read_payload

        def read(expert):
            entered.set()
            if not release.wait(5):
                raise TimeoutError('Test did not release read')
            return original(expert)

        try:
            with patch.object(bank, 'read_payload', side_effect=read):
                self.prefetch.begin(self.priority)
                self.assertTrue(entered.wait(5))
                self.prefetch.advance(3)
                self.assertIn((3, 100), self.prefetch.worker.pending)
                self.assertNotIn((3, 101), self.prefetch.worker.pending)
                self.assertEqual(bank.slots, {})
                release.set()
                self.ready()
                self.prefetch.advance(4)
                self.assertNotIn((3, 100), self.prefetch.worker.pending)
                self.assertEqual(bank.stats['prefetch_discarded'], 1)
                self.assertEqual(bank.stats['prefetch_cancelled'], 1)
                self.assertEqual(bank.slots, {})
        finally:
            release.set()

    def test_malformed_payload_is_not_published(self):
        bank = self.banks[0]
        with patch.object(bank, 'read_payload', return_value=({'gate.word': bytearray(1)}, 1, 0.0)):
            self.prefetch.begin(self.priority)
            self.ready()
            with self.assertRaisesRegex(ValueError, 'malformed'):
                bank.select((100,))
        self.assertNotIn(100, bank.slots)
        self.assertEqual(len(bank.free_slots), 32)

    def test_running_required_read_waits_and_installs_only_on_main_thread(self):
        entered, release = threading.Event(), threading.Event()
        bank = self.banks[0]
        read, install = bank.read_payload, bank._install_payload
        threads = {}

        def delayed(expert):
            threads['read'] = threading.get_ident()
            entered.set()
            if not release.wait(5):
                raise TimeoutError('Test did not release worker')
            return read(expert)

        def publish(*args):
            threads['install'] = threading.get_ident()
            return install(*args)

        timer = threading.Timer(0.02, release.set)
        try:
            with patch.object(bank, 'read_payload', side_effect=delayed), patch.object(bank, '_install_payload', side_effect=publish):
                self.prefetch.begin(self.priority)
                self.assertTrue(entered.wait(5))
                timer.start()
                with patch.object(bank, '_fill', side_effect=AssertionError('Unexpected synchronous read')):
                    row = bank.select((100,))[0]
                actual = bank.planes['gate'][0]['word'][row]
                bank.retain(actual)
                self.assertTrue(mx.array_equal(actual, self.weights[100]).item())
            self.assertNotEqual(threads['read'], threading.get_ident())
            self.assertEqual(threads['install'], threading.get_ident())
            self.assertGreater(bank.stats['prefetch_wait_seconds'], 0)
            self.assertEqual(bank.stats['prefetch_hits'], 1)
        finally:
            release.set()
            timer.join(5)

    def test_saturation_restart_and_zero_priority_leave_no_staging(self):
        self.prefetch.candidates = 4
        self.prefetch.begin(np.ones((42, 288)))
        self.assertEqual(len(self.prefetch.worker.pending), 8)
        self.assertEqual({key[0] for key in self.prefetch.worker.pending}, {3, 4})
        self.assertEqual(self.prefetch.submit_refused, 1)
        self.prefetch.begin(np.zeros((42, 288)))
        self.assertEqual(self.prefetch.worker.reserved_bytes, 0)
        self.prefetch.advance(44)
        self.assertEqual(self.prefetch.worker.pending, {})
        completed = sum(bank.stats['prefetch_hits'] + bank.stats['prefetch_discarded']
                        + bank.stats['prefetch_cancelled'] + bank.stats['prefetch_errors'] for bank in self.banks)
        self.assertEqual(completed, self.prefetch.submitted)
        self.assertGreater(self.prefetch.drain_seconds, 0)

    def test_allocation_error_is_fatal_and_does_not_publish(self):
        self.prefetch.candidates = 1
        bank = self.banks[0]
        with patch.object(bank, 'read_payload', side_effect=MemoryError('allocation failed')):
            self.prefetch.begin(self.priority)
            future = self.prefetch.worker.pending[3, 100][0]
            with self.assertRaises(MemoryError):
                future.result(timeout=5)
            with self.assertRaisesRegex(MemoryError, 'allocation failed'):
                bank.select((100,))
        self.assertEqual(bank.slots, {})
        self.assertEqual(len(bank.free_slots), 32)


if __name__ == '__main__':
    unittest.main()
