"""CPU regressions for uptime-dependent scheduling and oplog retention. No torch/GPU imports."""
import ast
import collections
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'streaming'))
import oplog
from scheduler_tap import TapScheduler


class TapLiveness(unittest.TestCase):
    def make(self):
        self.clock = 0.
        self.pred = types.SimpleNamespace(
            S=np.array([[64., 0., 0., 0.]], np.float32),
            step=lambda *a, **k: True,
        )
        env = dict(NQ_TAP_TODO_FIX='2', NQ_TAP_QREAL='1', NQ_TAP_RATE_FLOOR_GBPS='0',
                   NQ_TAP_H='256', NQ_TAP_HA='0', NQ_TAP_MLA='1', NQ_TAP_LAT='model',
                   NQ_TAP_RATE_GBPS='6', NQ_TAP_PEAK_GBPS='0', NQ_TAP_CTL='/nonexistent/audit')
        with patch.dict(os.environ, env):
            s = TapScheduler([3], {3: []}, {3: [0]}, rec_bytes=40_000_000,
                             NE=4, n_float=1, slots=2, predictor=self.pred,
                             hostloop='py', clock=lambda: self.clock)
        s.state[0, 0] = 2
        s.xq = lambda: 0
        return s

    def test_idle_then_new_request_keeps_upgrade_capacity(self):
        s = self.make()
        for _ in range(8000):
            self.clock += .16
            self.assertEqual(s.step(np.array([[128., 0., 0., 0.]]), 16), ([], []))
        self.assertAlmostEqual(s.peak[0], s.rate0)
        self.pred.S[:] = [[0., 64., 0., 0.]]
        self.clock += .16
        ups, downs = s.step(np.array([[0., 128., 0., 0.]]), 16, new_request=True)
        self.assertEqual(ups, [(3, 1)])
        self.assertEqual(downs, [])

    def test_follower_backpressure_is_not_bypassed(self):
        s = self.make()
        s.tps = 100.
        s.io_all = lambda: {0: {}, 1: dict(ops_outstanding=100000, delivered_GBps=0.)}
        _, _, budget = s._plan(1., 0, 0)
        self.assertEqual(budget, 0)
        p = s.peak.copy()
        s._plan(2., 0, 0)
        self.assertEqual(s.peak[0], p[0])  # idle leader preserves its rate
        self.assertLess(s.peak[1], p[1])  # busy follower still drives backpressure

    def test_busy_rank_recovers_when_a_read_lands(self):
        s = self.make()
        s.tps = 100.
        s.peak = np.array([.00001])
        s._plan(0., 1, 0)
        s.n_land += 1
        _, _, budget = s._plan(.3, 0, 0)
        self.assertGreater(s.peak[0], 3.)
        self.assertGreater(budget, 0)

    def test_idle_follower_estimate_does_not_decay(self):
        s = self.make()
        s.io_all = lambda: {0: {}, 1: dict(ops_outstanding=0, slot_wait=0, backlog=0, delivered_GBps=0.)}
        for i in range(10000):
            s._plan(float(i), 0, 0)
        np.testing.assert_array_equal(s.peak, [s.rate0, s.rate0])


class LogRetention(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = self.tmp.name + '/ops'
        self.oldrot = oplog.ROT
        oplog.ROT = 32
        self.logs = []

    def tearDown(self):
        for log in self.logs:
            log.close()
        oplog.ROT = self.oldrot
        self.tmp.cleanup()

    def log(self, writer, **kw):
        x = oplog.OpLog(self.path, writer, **kw)
        self.logs.append(x)
        return x

    def test_delayed_and_not_started_followers_keep_every_record(self):
        w = self.log(True, reader_count=2)
        fast = self.log(False, reader_id=1, reader_count=2)
        expected = []
        got = []
        for e in range(20):
            row = ([(3, e)], [])
            w.put(*row)
            expected.append(row)
            got.extend(fast.get())
        self.assertTrue(Path(w._p(0)).exists())  # follower2 not even registered yet
        slow = self.log(False, reader_id=2, reader_count=2)
        self.assertEqual(slow.get(), expected)
        self.assertEqual(got, expected)
        w._gc()
        self.assertFalse(Path(w._p(0)).exists())
        self.assertTrue(Path(w._p(w.gen)).exists())
        self.assertTrue(Path(w._p(w.gen - 1)).exists())

    def test_open_reader_can_lag_many_rotations(self):
        w = self.log(True, reader_count=1)
        r = self.log(False, reader_id=1, reader_count=1)
        w.put([(3, 0)], [])
        self.assertEqual(r.get(), [([(3, 0)], [])])
        for e in range(1, 30):
            w.put([(3, e)], [])
        self.assertEqual(r.get(), [([(3, e)], []) for e in range(1, 30)])

    def test_unknown_membership_retains_history(self):
        w = self.log(True)
        for e in range(8):
            w.put([(3, e)], [])
        self.assertTrue(Path(w._p(0)).exists())

    def test_missing_generation_is_an_explicit_error(self):
        w = self.log(True, reader_count=1)
        for e in range(8):
            w.put([(3, e)], [])
        Path(w._p(0)).unlink()  # simulate external loss; never silently skip unknown expert decisions
        r = self.log(False, reader_id=1, reader_count=1)
        with self.assertRaisesRegex(RuntimeError, 'resynchronization required'):
            r.get()

    def test_get_into_acknowledges_only_consumed_generations(self):
        w = self.log(True, reader_count=1)
        for e in range(8):
            w.put([(3, e)], [])
        r = self.log(False, reader_id=1, reader_count=1)
        seen = []
        def feed(a):
            i = 0
            while i + 3 <= len(a):
                self.assertEqual(a[i], oplog.MAGIC)
                nu, nd = int(a[i + 1]), int(a[i + 2])
                if nu == -1:
                    return i + 3, True
                j = i + 3 + 2 * (nu + nd)
                if j > len(a):
                    break
                seen.extend(a[i + 3:j].reshape(-1, 2).tolist())
                i = j
            return i, False
        r.get_into(feed)
        self.assertEqual(seen, [[3, e] for e in range(8)])
        self.assertEqual(int(Path(self.path + '.ack.1').read_text()), w.gen)

    def test_reclaim_gate_pins_old_generation(self):
        w = self.log(True, reader_count=1)
        w.put_reclaim(7)
        for e in range(8):
            w.put([(3, e)], [])
        r = self.log(False, reader_id=1, reader_count=1)
        gate = types.SimpleNamespace(x_done=6, x_have=0)
        self.assertEqual(r.get(gate), [])
        w._gc()
        self.assertTrue(Path(w._p(0)).exists())
        gate.x_done = 7
        self.assertEqual(len(r.get(gate)), 8)


class LatencyRetention(unittest.TestCase):
    def test_actual_poll_keeps_only_recent_history(self):
        # Compile only poll to avoid importing torch or constructing GPU objects.
        tree = ast.parse((ROOT / 'streaming/executor.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RankExecutor')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'poll')
        ns = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), 'executor-poll', 'exec'), ns)
        s = types.SimpleNamespace(ops={}, odst={}, up_tag={}, layers={3: (None, types.SimpleNamespace(hseq={}), None)},
                                  wait_apply={}, lat=[], lat_w=collections.deque(maxlen=4096),
                                  lat_rc=collections.deque(maxlen=4096), n_landed=0, ah={3: {0: -1}}, pend=[], xpend=[])
        for j in range(20000):
            s.ops[j] = (3, 0, 4, j)
            s.eng = types.SimpleNamespace(poll=lambda j=j: [(j, True, .01, float(j))])
            ns['poll'](s, issue=False)
        self.assertEqual(len(s.lat), 4096)
        self.assertEqual(s.lat[-256:], list(map(float, range(19744, 20000))))
        self.assertEqual(s.n_landed, 20000)


if __name__ == '__main__':
    unittest.main()
