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
        keys = ('NQ_FLASH_JT_PARAMS', 'NQ_HOT_EVICT', 'NQ_HOT_VERIFY_WAIT', 'NQ_REFINEMENT_PENDING',
                'NQ_REFINEMENT_DIRECT', 'NQ_DIRECT_INFLIGHT', 'NQ_EVICT_AHEAD')
        with mock.patch.dict(os.environ, {}, clear=False):
            for key in keys:
                os.environ.pop(key, None)
            self.assertEqual(settings(), dict(jt=hot_policy.DEFAULTS['jt'], evict='score', verify_wait=False,
                                              pending=64, delivery='spare', direct_inflight=128, evict_ahead=2))
            os.environ.update(NQ_FLASH_JT_PARAMS=json.dumps(dict(hm=4.0, half_life=64.0)), NQ_HOT_EVICT='fifo',
                              NQ_HOT_VERIFY_WAIT='0', NQ_REFINEMENT_PENDING='48')
            current = settings()
            self.assertEqual(current['jt'], dict(mix=0.75, hm=4.0, G=1, half_life=64.0))
            self.assertEqual((current['evict'], current['verify_wait'], current['pending']), ('fifo', False, 48))
            self.assertEqual(json.loads(hot_policy.jt_parameters_json()), current['jt'])
            for raw, delivery in (('0', 'staged'), ('1', 'direct'), ('spare', 'spare'), ('staged', 'staged')):
                os.environ['NQ_REFINEMENT_DIRECT'] = raw
                self.assertEqual(settings()['delivery'], delivery)
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


class SpareDeliveryTest(unittest.TestCase):
    def bank(self, rows, resident):
        from packed_expert_bank import PackedRefinementBank
        bank = object.__new__(PackedRefinementBank)
        bank.slots = {expert: slot for slot, expert in enumerate(resident)}
        bank.table = np.full(288, PackedRefinementBank.COLD_SLOT, dtype=np.uint32)
        for expert, slot in bank.slots.items():
            bank.table[expert] = slot
        bank._device_table = None
        bank.previous_output = None
        bank.available = list(reversed(range(len(resident), rows)))
        return bank

    def hot(self, bank):
        from native_refinements import NativeHotRefinements
        hot = object.__new__(NativeHotRefinements)
        hot.reads = dict(installed=0, decode_bytes=0, decode_worker_seconds=0.0)
        hot.record = None
        hot.banks = {3: bank}
        hot.extra_rows = {3: 1}
        hot.evict_ahead = 1
        hot.evict = 'score'
        hot.spare_reads = True
        hot.direct_inflight = 8
        hot.install_threads = 2
        hot._copies = []
        hot._copy_pool = None
        hot.wanted = np.zeros((42, 288), dtype=bool)
        hot.priority = np.zeros((42, 288))
        hot.reader = mock.Mock()
        hot.reader.read_into_slot.return_value = (100, 0.001)
        return hot

    def test_take_free_never_evicts(self):
        bank = self.bank(3, [10, 11])
        self.assertEqual(bank.take_free(12), 2)
        self.assertEqual(bank.staging, {12: 2})
        self.assertIsNone(bank.take_free(13))
        self.assertIsNone(bank.take_free(10))
        bank.evict(11)
        self.assertNotIn(11, bank.slots)
        self.assertEqual(bank.table[11], bank.COLD_SLOT)
        self.assertEqual(bank.take_free(13), 1)

    def test_reads_go_to_free_slots_and_activate_at_flip(self):
        bank = self.bank(4, [10, 11, 12])          # pool of 3 plus one spare slot
        hot = self.hot(bank)
        hot.wanted[0, [10, 20, 21]] = True
        hot.priority[0, [10, 11, 12, 20, 21]] = [.9, .1, .2, .8, .7]
        hot.order = [(3, 20), (3, 21)]
        hot._queue_spare()                          # only the spare slot is free
        self.assertEqual(list(bank.staging), [20])
        hot.flip(wait=True)
        self.assertIn(20, bank.slots)
        self.assertEqual(hot.reads['installed'], 1)
        # publish: free the spare slot again (the still-missing 21 fits it; evict_ahead covers only misses
        # beyond the spare slots), evicting the lowest-scoring expert outside the wanted set.
        hot._evict_for_reads()
        self.assertEqual(len(bank.available), 1)
        self.assertNotIn(11, bank.slots)
        self.assertIn(12, bank.slots)
        hot.wanted[0, 22] = True                    # two misses, one spare: evict one more ahead
        hot.priority[0, 22] = .6
        hot._evict_for_reads()
        self.assertEqual(len(bank.available), 2)
        self.assertNotIn(12, bank.slots)
        hot.order = [(3, 21), (3, 22)]
        hot._queue_spare()
        hot.flip(wait=True)
        self.assertIn(21, bank.slots)
        self.assertIn(22, bank.slots)
        hot._copy_pool.shutdown()
