import threading
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from experts import ExpertStore
from metrics import MemoryGuard
from prefetch import BoundedReader


class BoundedReaderTests(unittest.TestCase):
    def test_slow_read_is_nonblocking_and_staging_is_bounded(self):
        entered, release = threading.Event(), threading.Event()
        def read(*key):
            entered.set()
            if not release.wait(5):
                raise TimeoutError('Test did not release worker')
            return key
        queue = BoundedReader(read, 2 * (3 * 8 + 1024**2))
        try:
            self.assertTrue(queue.submit((3, 0), 8))
            self.assertTrue(entered.wait(5))
            self.assertTrue(queue.submit((3, 1), 8))
            self.assertFalse(queue.submit((3, 2), 8))
            self.assertFalse(queue.submit((3, 0), 8))
            values = []
            queue.poll(lambda *item: values.append(item))
            self.assertEqual(values, [])
            self.assertLessEqual(queue.reserved_bytes, queue.budget_bytes)
            release.set()
            for future, _ in queue.pending.values():
                future.result(timeout=5)
            queue.poll(lambda *item: values.append(item))
            self.assertEqual([key for key, _ in values], [(3, 0), (3, 1)])
            self.assertEqual(queue.reserved_bytes, 0)
        finally:
            release.set()
            queue.close()
        with self.assertRaises(RuntimeError):
            queue.submit((3, 0), 8)

    def test_worker_failure_propagates_and_releases_reservation(self):
        def read(*key):
            raise OSError('SSD read failed')
        queue = BoundedReader(read, 2 * 1024**2)
        queue.submit((3, 0), 8)
        future = queue.pending[3, 0][0]
        with self.assertRaises(OSError):
            future.result(timeout=5)
        with self.assertRaisesRegex(OSError, 'SSD read failed'):
            queue.poll(lambda *_: None)
        self.assertEqual(queue.reserved_bytes, 0)
        queue.close()
        queue.close()

    def test_close_joins_reader_before_releasing_storage(self):
        entered, release, finished = threading.Event(), threading.Event(), threading.Event()
        def read(*key):
            entered.set()
            release.wait(5)
            finished.set()
            return key
        queue = BoundedReader(read, 2 * 1024**2)
        queue.submit((3, 0), 8)
        self.assertTrue(entered.wait(5))
        closer = threading.Thread(target=queue.close)
        closer.start()
        self.assertFalse(finished.is_set())
        release.set()
        closer.join(5)
        self.assertFalse(closer.is_alive())
        self.assertTrue(finished.is_set())
        self.assertEqual(queue.reserved_bytes, 0)


class AsyncStoreTests(unittest.TestCase):
    def setUp(self):
        self.entered, self.release = threading.Event(), threading.Event()
        self.store = ExpertStore(Path('.'), cache_gib=0.1, resident_base=True, async_refinements=True)
        self.store.base_cache.update({(3, expert): {} for expert in (0, 1, 7)})
        def read(layer, expert):
            self.entered.set()
            if not self.release.wait(5):
                raise TimeoutError('Test did not release worker')
            return {'gate': {'p4': np.zeros(8, dtype=np.uint8)}}, 8, 0.01
        self.store.prefetch.read = read
        self.store.refinement_bytes = lambda *_: 8
        self.installed_threads = []
        def install(layer, expert, payload):
            self.installed_threads.append(threading.get_ident())
            return payload
        self.store._materialize = install
        self.budgets = {layer: 1 for layer in range(3, 45)}
        self.priorities = np.ones((42, 288))

    def tearDown(self):
        self.release.set()
        self.store.close()

    def wanted(self, expert):
        result = np.zeros((42, 288), dtype=bool)
        result[0, expert] = True
        return result

    def ready(self):
        self.release.set()
        for future, _ in self.store.prefetch.pending.values():
            future.result(timeout=5)
        self.store.poll_refinements()

    def test_pending_reads_do_not_replace_residents_until_ready(self):
        self.store.cache[3, 7, 4] = {'old': True}, 8
        self.store.cache_bytes = 8
        self.store.begin_decode(self.wanted(0), self.priorities, self.budgets)
        self.assertTrue(self.entered.wait(5))
        self.assertIn((3, 7, 4), self.store.cache)
        self.assertNotIn((3, 0, 4), self.store.cache)
        self.ready()
        self.assertEqual(set(self.store.cache), {(3, 0, 4)})
        self.assertEqual(self.installed_threads, [threading.get_ident()])
        self.assertEqual(self.store.cache_bytes, 8)
        self.assertLessEqual(self.store.cache_bytes + self.store.staging_budget, self.store.cache_budget)
        self.assertEqual(self.store.metrics['async_loaded'], 1)

    def test_stale_completion_is_discarded_after_prediction_changes(self):
        self.store.begin_decode(self.wanted(0), self.priorities, self.budgets)
        self.assertTrue(self.entered.wait(5))
        self.store.begin_decode(self.wanted(1), self.priorities, self.budgets)
        self.ready()
        self.assertNotIn((3, 0, 4), self.store.cache)
        self.assertIn((3, 1, 4), self.store.cache)
        self.assertEqual(self.store.metrics['async_discarded'], 1)

    def test_reset_drains_pending_io_without_installing(self):
        self.store.begin_decode(self.wanted(0), self.priorities, self.budgets)
        self.release.set()
        self.store.finish_decode()
        self.assertFalse(self.store.prefetch.pending)
        self.assertFalse(self.store.cache)
        self.assertEqual(self.store.prefetch.reserved_bytes, 0)
        self.assertFalse(self.installed_threads)

    def test_completion_for_evicted_base_is_discarded(self):
        self.store.begin_decode(self.wanted(0), self.priorities, self.budgets)
        self.assertTrue(self.entered.wait(5))
        del self.store.base_cache[3, 0]
        self.ready()
        self.assertFalse(self.store.cache)
        self.assertFalse(self.store.prefetch.pending)
        self.assertFalse(self.installed_threads)
        self.assertEqual(self.store.metrics['async_discarded'], 1)

    def test_missing_base_does_not_schedule_refinement(self):
        del self.store.base_cache[3, 0]
        self.store.begin_decode(self.wanted(0), self.priorities, self.budgets)
        self.assertFalse(self.store.prefetch.pending)


class BaseCacheTests(unittest.TestCase):
    def test_base_eviction_removes_its_refinement_and_honors_recency(self):
        store = ExpertStore(Path('.'), resident_base=True, base_cache_gib=16 / 1024**3)
        store._load = lambda *args: {'gate': ({'base': np.zeros(8, dtype=np.uint8)}, {})}
        try:
            store.get(3, 0, 2)
            store.get(3, 1, 2)
            store.get(3, 0, 2)
            store.cache[3, 1, 4] = {'refined': True}, 4
            store.cache_bytes = 4
            store.get(3, 2, 2)
            self.assertEqual(set(store.base_cache), {(3, 0), (3, 2)})
            self.assertEqual(store.base_bytes, 16)
            self.assertFalse(store.cache)
            self.assertEqual(store.cache_bytes, 0)
            self.assertEqual(store.metrics['base_evictions'], 1)
            with self.assertRaisesRegex(ValueError, 'on demand'):
                store.preload_base()
        finally:
            store.close()
        self.assertFalse(store.base_sizes)

    def test_refinement_hit_keeps_its_base_recent(self):
        store = ExpertStore(Path('.'), resident_base=True, base_cache_gib=16 / 1024**3)
        store._load = lambda *args: {'gate': ({'base': np.zeros(8, dtype=np.uint8)}, {})}
        try:
            store.get(3, 0, 2)
            store.get(3, 1, 2)
            store.cache[3, 0, 4] = {'refined': True}, 4
            store.cache_bytes = 4
            store.get(3, 0, 4)
            store.get(3, 2, 2)
            self.assertIn((3, 0), store.base_cache)
            self.assertIn((3, 0, 4), store.cache)
            self.assertNotIn((3, 1), store.base_cache)
        finally:
            store.close()


class MemoryGuardTests(unittest.TestCase):
    def test_native_resource_probes(self):
        from resources import process_usage, system_pressure
        self.assertGreater(process_usage().phys_footprint, 0)
        self.assertIn(system_pressure(), (1, 2, 4))

    def test_footprint_and_system_pressure_fail_closed(self):
        with patch('metrics.mx.device_info', return_value={'memory_size': 128 * 1024**3}), \
             patch('metrics.process_usage', return_value=SimpleNamespace(phys_footprint=1)) as usage, \
             patch('metrics.system_pressure', return_value=1) as pressure:
            guard = MemoryGuard(84)
            usage.return_value.phys_footprint = 85 * 1024**3
            with self.assertRaisesRegex(MemoryError, 'exceeds'):
                guard.check(force=True)
            usage.return_value.phys_footprint = 1
            pressure.return_value = 2
            with self.assertRaisesRegex(MemoryError, 'memory pressure'):
                guard.check(force=True)
