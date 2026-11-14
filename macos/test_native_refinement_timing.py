from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from native_refinements import COLD_SLOT, NativeHotRefinements


class NativeRefinementTimingTest(unittest.TestCase):
    def runtime(self, phase):
        hot = NativeHotRefinements.__new__(NativeHotRefinements)
        hot.worker = Mock()
        hot.predictor = SimpleNamespace(reset=Mock(), budgets={layer: 2 for layer in range(3, 45)},
                                        defaults=[], fixed_ids=[])
        with patch('native_refinements.PredictorMetrics'):
            hot.reset()
        hot.phase = phase
        hot._queue = Mock()
        hot.reader = Mock()
        hot.prefill_layer = 3
        hot.prefill_bank = Mock(slots={1: 0})
        hot.banks = {3: Mock(slots={1: 0})}
        return hot

    def test_prefill_times_only_missing_rows_and_keeps_reader_separate(self):
        hot = self.runtime('prefill')
        payload = object()
        hot.reader.read.return_value = payload, 128, 2.5

        def install(expert, value, required):
            self.assertIs(value, payload)
            self.assertEqual(required, {1, 2})
            hot.prefill_bank.slots[expert] = 1
            return True

        hot.prefill_bank.install.side_effect = install
        with patch('native_refinements.time.perf_counter', side_effect=[10.0, 10.25]), \
                patch('native_refinements.mx.array', side_effect=lambda value, **_: value):
            bank, slots = hot.route(3, None, (1, 2, 2), None, None)
        self.assertIs(bank, hot.prefill_bank)
        self.assertEqual(slots, [0, 1, 1])
        hot.reader.read.assert_called_once_with(3, 2)
        self.assertEqual(hot.reads['prefill_bytes'], 128)
        self.assertEqual(hot.reads['prefill_reader_seconds'], 2.5)
        self.assertEqual(hot.reads['prefill_install_seconds'], 0.25)
        self.assertEqual(hot.reads['decode_publication_seconds'], 0)

    def test_decode_times_publication_without_adding_worker_time(self):
        hot = self.runtime('decode')
        hot.wanted[0, 2] = True
        payload = object()

        def poll(publish):
            publish((3, 2), (payload, 256, 1.5))
            publish((3, 9), (object(), 512, 2.0))

        def install(expert, value, wanted, scores=None):
            self.assertIs(value, payload)
            self.assertEqual(wanted, {2})
            hot.banks[3].slots[expert] = 1
            return True

        hot.worker.poll.side_effect = poll
        hot.banks[3].install.side_effect = install
        with patch('native_refinements.time.perf_counter', side_effect=[20.0, 20.125, 21.0, 21.25]), \
                patch('native_refinements.mx.array', side_effect=lambda value, **_: value):
            bank, slots = hot.route(3, None, (2, 9), None, None)
        self.assertIs(bank, hot.banks[3])
        self.assertEqual(slots, [1, COLD_SLOT])
        self.assertEqual(hot.reads['decode_bytes'], 768)
        self.assertEqual(hot.reads['decode_worker_seconds'], 3.5)
        self.assertEqual(hot.reads['decode_publication_seconds'], 0.375)
        self.assertEqual(hot.reads['prefill_reader_seconds'], 0)
        self.assertEqual(hot.reads['prefill_install_seconds'], 0)
        self.assertEqual(hot.reads['installed'], 1)
        self.assertEqual(hot.reads['discarded'], 1)
        hot.banks[3].install.assert_called_once()
        hot._queue.assert_called_once_with()
        self.assertTrue(hot.ready[0, 2])
        self.assertFalse(hot.ready[0, 9])

    def test_rejected_required_install_still_fails(self):
        hot = self.runtime('prefill')
        hot.reader.read.return_value = {}, 128, 2.5
        hot.prefill_bank.install.return_value = False
        with patch('native_refinements.time.perf_counter', side_effect=[10.0, 10.25]):
            with self.assertRaisesRegex(RuntimeError, 'Required prefill refinement'):
                hot.route(3, None, (2,), None, None)
        self.assertNotIn(2, hot.prefill_bank.slots)
        self.assertEqual(hot.reads['prefill_install_seconds'], 0.25)
        self.assertEqual(hot.reads['prefill_reader_seconds'], 2.5)

    def test_reset_clears_timing_and_drains_worker(self):
        hot = self.runtime('decode')
        hot.reads.update(prefill_reader_seconds=10, prefill_install_seconds=20, decode_publication_seconds=30)
        hot.worker.reset_mock()
        with patch('native_refinements.PredictorMetrics'):
            hot.reset()
        hot.worker.discard.assert_called_once_with()
        for name in ('prefill_reader_seconds', 'prefill_install_seconds', 'decode_publication_seconds'):
            self.assertEqual(hot.reads[name], 0)
        self.assertEqual(hot.phase, 'prefill')


if __name__ == '__main__':
    unittest.main()
