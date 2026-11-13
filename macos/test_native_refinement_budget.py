from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from native_refinements import NativeHotRefinements


GIB = 1024**3


class RefinementBudgetTest(unittest.TestCase):
    def test_invalid_budget_fails_before_predictor_or_worker_allocation(self):
        for budget in (None, 0, 3, 33, True, 4.0, float('inf')):
            with self.subTest(budget=budget), patch('native_refinements.staged_predictor') as staged:
                with self.assertRaisesRegex(ValueError, 'between 4 and 32 GiB'):
                    NativeHotRefinements(None, .05, 128, budget_gib=budget)
                staged.assert_not_called()

    def test_default_and_explicit_budget_are_recorded(self):
        for budget in (None, 4, 6):
            with self.subTest(budget=budget), \
                    patch('native_refinements.staged_predictor', return_value=(Mock(), Mock())), \
                    patch('native_refinements.configure_slots'), \
                    patch('native_refinements.RefinementReader'), \
                    patch('native_refinements.BoundedReader'), \
                    patch.object(NativeHotRefinements, 'reset'):
                arguments = {} if budget is None else dict(budget_gib=budget)
                hot = NativeHotRefinements(None, .05, 128, **arguments)
                self.assertEqual(hot.budget_bytes, (4 if budget is None else budget) * GIB)

    def runtime(self, budget, existing):
        hot = NativeHotRefinements.__new__(NativeHotRefinements)
        hot.budget_bytes = budget
        hot.prefill_rows = 128
        hot.prefill_max_bytes = GIB
        hot.worker = SimpleNamespace(budget_bytes=64 * 1024**2)
        hot.banks = {3: SimpleNamespace(bytes=existing)}
        hot.predictor = SimpleNamespace(budgets={4: 16})
        return hot

    def test_default_rejects_allocation_that_six_gib_can_hold(self):
        for budget, accepted in ((4, False), (6, True)):
            hot = self.runtime(budget * GIB, 2 * GIB)
            with self.subTest(budget=budget), patch('native_refinements.PackedRefinementBank') as bank:
                bank.bytes_for.side_effect = (2 * GIB, GIB)
                if accepted:
                    hot.attach(4, None)
                    bank.assert_called_once_with(None, 16)
                    self.assertIs(hot.banks[4], bank.return_value)
                else:
                    with self.assertRaisesRegex(MemoryError, 'refinement budget'):
                        hot.attach(4, None)
                    bank.assert_not_called()
                    self.assertEqual(set(hot.banks), {3})

    def test_boundary_counts_decode_scratch_and_worker_before_allocation(self):
        existing = 6 * GIB - 2 * GIB - GIB - 64 * 1024**2
        for extra in (0, 1):
            hot = self.runtime(6 * GIB, existing + extra)
            with self.subTest(extra=extra), patch('native_refinements.PackedRefinementBank') as bank:
                bank.bytes_for.side_effect = (2 * GIB, GIB)
                if extra:
                    with self.assertRaises(MemoryError):
                        hot.attach(4, None)
                    bank.assert_not_called()
                else:
                    hot.attach(4, None)
                    bank.assert_called_once_with(None, 16)


if __name__ == '__main__':
    unittest.main()
