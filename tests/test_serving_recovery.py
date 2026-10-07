"""CPU-only failure injection; extract real methods without importing serving code."""
import ast
import collections
import copy
import logging
from pathlib import Path
import queue
import types
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / 'sm120' / 'serve'
LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())
LOG.propagate = False


def methods(file, cls, names, env):
    node = next(n for n in ast.parse((ROOT/file).read_text()).body
                if isinstance(n, ast.ClassDef) and n.name == cls)
    node.body = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in names]
    ns = {'log': LOG, **env}
    exec(compile(ast.Module(body=[node], type_ignores=[]), file, 'exec'), ns)
    return ns[cls]


class CV:
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def wait_for(self, *args, **kwargs): return False
    def notify_all(self): pass


class RecoveryTests(unittest.TestCase):
    def test_pb_timeout_restores_counter_and_raises(self):
        C = methods('nq_pb.py', 'PB', {'_pause'}, {})
        obj = object.__new__(C)
        obj.rt = types.SimpleNamespace(cv=CV(), pb_pause=2, in_iter=True)
        with self.assertRaises(TimeoutError): obj._pause()
        self.assertEqual(obj.rt.pb_pause, 2)

    def test_lmpf_timeout_never_snapshots(self):
        C = methods('nq_lmpf.py', 'LM', {'pause'}, {'PAUSE_S': 0})
        obj = object.__new__(C)
        obj.paused = False
        obj.rt = types.SimpleNamespace(cv=CV(), ncap=3, X=object())
        obj.snap_tables = lambda: self.fail('unsafe snapshot')
        with self.assertRaises(TimeoutError): obj.pause()
        self.assertEqual(obj.rt.ncap, 3)
        self.assertFalse(obj.paused)

    def test_ring_failure_never_becomes_silent_fallback(self):
        C = methods('nq_lmpf.py', 'LM', {'ring_off'}, {})
        for close_fails in (False, True):
            obj = object.__new__(C)
            def close():
                if close_fails: raise OSError('I/O synchronization failed')
            ring = types.SimpleNamespace(eng=types.SimpleNamespace(close=close))
            obj.ring = ring
            obj.rt = types.SimpleNamespace(rank=0)
            with self.assertRaises((OSError, RuntimeError)): obj.ring_off('read failure')
            self.assertIs(obj.ring, ring if close_fails else None)

    def test_session_schema_and_identity(self):
        tree = ast.parse((ROOT/'nq_session.py').read_text())
        nodes = [n for n in tree.body if
                 (isinstance(n, ast.FunctionDef) and n.name in
                  {'_check_predictor', '_restore_value', 'p_snap', 'p_load'}) or
                 (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'PSTATE' for t in n.targets))]
        ns = {'np': np, 'copy': copy}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), 'nq_session.py', 'exec'), ns)
        # Stand-in has the exact supported schema; no LightGBM or GPU required.
        C = type('GBDTPredictor', (), {'__module__': 'gbdt_predictor'})
        p = C(); p.E = [np.array([1., 2.])]; p.S = np.array([3., 4.])
        p._res = queue.Queue(); p._res.put(np.array([5.])); p._pending = True
        original_e, original_s = p.E[0], p.S
        snap = ns['p_snap'](p)
        p.E[0][:] = 9; p.S[:] = 8
        ns['p_load'](p, snap)
        self.assertIs(p.E[0], original_e); self.assertIs(p.S, original_s)
        np.testing.assert_array_equal(p.E[0], [1., 2.])
        np.testing.assert_array_equal(p.S, [3., 4.])
        self.assertTrue(p._pending)
        np.testing.assert_array_equal(p._res.get_nowait(), [5.])
        bad = types.SimpleNamespace(_pending=True)
        with self.assertRaisesRegex(RuntimeError, 'unsupported predictor'): ns['p_snap'](bad)
        self.assertTrue(bad._pending)  # rejected before consuming pending work
        with self.assertRaisesRegex(RuntimeError, 'unsupported predictor'): ns['p_load'](bad, snap)


if __name__ == '__main__': unittest.main()
