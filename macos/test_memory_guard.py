import unittest
from unittest import mock

import metrics


class PressurePolicyTest(unittest.TestCase):
    def guard(self, **attributes):
        with mock.patch.object(metrics, 'system_pressure', return_value=1):
            guard = metrics.MemoryGuard(84)
        for name, value in attributes.items():
            setattr(guard, name, value)
        return guard

    def test_default_guard_stops_at_warning_pressure(self):
        guard = self.guard()
        with mock.patch.object(metrics, 'system_pressure', return_value=2):
            with self.assertRaises(MemoryError):
                guard.check(force=True)

    def test_externally_watched_guard_tolerates_warning_pressure(self):
        guard = self.guard(allowed_pressure=(1, 2))
        with mock.patch.object(metrics, 'system_pressure', return_value=2):
            guard.check(force=True)

    def test_externally_watched_guard_still_stops_at_critical_pressure(self):
        guard = self.guard(allowed_pressure=(1, 2))
        with mock.patch.object(metrics, 'system_pressure', return_value=4):
            with self.assertRaises(MemoryError):
                guard.check(force=True)


if __name__ == '__main__':
    unittest.main()
