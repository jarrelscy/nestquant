import json
import os
import threading
import unittest
from concurrent.futures import Future
from unittest import mock

import numpy as np

import hot_policy
from hot_policy import choose_victim, settings


class ChooseVictimTest(unittest.TestCase):
    def test_fifo_takes_oldest_install_outside_wanted(self):
        slots = {5: 0, 7: 1, 9: 2}
        self.assertEqual(choose_victim(slots, {5}), 7)
        self.assertIsNone(choose_victim(slots, {5, 7, 9}))

    def test_score_takes_lowest_priority_outside_wanted(self):
        slots = {5: 0, 7: 1, 9: 2, 11: 3}
        scores = np.zeros(288)
        scores[[5, 7, 9, 11]] = [0.0, 0.3, 0.1, 0.2]
        self.assertEqual(choose_victim(slots, {5}, scores), 9)
        self.assertEqual(choose_victim(slots, {5, 9}, scores), 11)
        self.assertIsNone(choose_victim(slots, {5, 7, 9, 11}, scores))

    def test_score_ties_keep_insertion_order(self):
        slots = {9: 0, 7: 1, 5: 2}
        self.assertEqual(choose_victim(slots, set(), np.zeros(288)), 9)


class SettingsTest(unittest.TestCase):
    def test_defaults_and_overrides(self):
        keys = ('NQ_FLASH_JT_PARAMS', 'NQ_HOT_EVICT', 'NQ_HOT_VERIFY_WAIT', 'NQ_REFINEMENT_PENDING')
        with mock.patch.dict(os.environ, {}, clear=False):
            for key in keys:
                os.environ.pop(key, None)
            self.assertEqual(settings(), dict(jt=hot_policy.DEFAULTS['jt'], evict='score', verify_wait=False,
                                              pending=64))
            os.environ.update(NQ_FLASH_JT_PARAMS=json.dumps(dict(hm=4.0, half_life=64.0)), NQ_HOT_EVICT='fifo',
                              NQ_HOT_VERIFY_WAIT='0', NQ_REFINEMENT_PENDING='48')
            current = settings()
            self.assertEqual(current['jt'], dict(mix=0.75, hm=4.0, G=1, half_life=64.0))
            self.assertEqual((current['evict'], current['verify_wait'], current['pending']), ('fifo', False, 48))
            self.assertEqual(json.loads(hot_policy.jt_parameters_json()), current['jt'])
            os.environ['NQ_HOT_EVICT'] = 'random'
            with self.assertRaises(ValueError):
                settings()

    def test_pending_fits_staging_budget(self):
        # 64 expert-major records (at most 8,339,456 bytes each) inside the 512 MiB decode staging budget.
        self.assertLessEqual(hot_policy.DEFAULTS['pending'] * 8_339_456, 512 * 1024**2)


class RuntimeHooksTest(unittest.TestCase):
    def hot(self):
        from native_refinements import NativeHotRefinements
        hot = object.__new__(NativeHotRefinements)
        hot.reads = dict(installed=0)
        hot.record = None
        hot.direct_reads = False
        return hot

    def test_victim_scores_follow_eviction_rule(self):
        hot = self.hot()
        hot.priority = np.arange(42 * 288, dtype=float).reshape(42, 288)
        hot.evict = 'score'
        np.testing.assert_array_equal(hot._victim_scores(5), hot.priority[2])
        hot.evict = 'fifo'
        self.assertIsNone(hot._victim_scores(5))

    def test_flip_for_verify_waits_for_started_copies(self):
        hot = self.hot()
        hot.verify_wait = True
        bank = mock.Mock()
        future = Future()
        hot._copies = [(bank, 7, 3, future)]
        timer = threading.Timer(0.05, future.set_result, (None,))
        timer.start()
        hot.flip_for_verify()
        timer.join()
        bank.activate.assert_called_once_with(7, 3)
        self.assertEqual(hot._copies, [])
        self.assertIn('verify_wait_seconds', hot.reads)

    def test_flip_for_verify_without_wait_leaves_running_copies(self):
        hot = self.hot()
        hot.verify_wait = False
        bank = mock.Mock()
        future = Future()
        hot._copies = [(bank, 7, 3, future)]
        hot.flip_for_verify()
        bank.activate.assert_not_called()
        self.assertEqual(len(hot._copies), 1)


if __name__ == '__main__':
    unittest.main()
