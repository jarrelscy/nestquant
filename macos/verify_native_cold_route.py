from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import mlx.core as mx

from native_generate import PagedExperts


class ColdRouteTest(unittest.TestCase):
    def run_route(self, enabled, ready, phase='decode'):
        ids = mx.array([[[2, 5]]])
        weights = mx.array([[[0.25, 0.75]]])
        planes = {name: ({}, {}) for name in ('gate', 'up', 'down')}
        bank = SimpleNamespace(fully_resident=False, layer=3, select=Mock(return_value=(0, 1)),
                               planes=planes, retain=Mock())
        hot_bank = SimpleNamespace(slots={expert: index for index, expert in enumerate(ready)},
                                   planes={name: {'fixture': name} for name in planes}, retain=Mock())
        hot_route = mx.array([hot_bank.slots.get(expert, 0xffffffff) for expert in (2, 5)], dtype=mx.uint32)
        hot = SimpleNamespace(phase=phase, route=Mock(return_value=(hot_bank, hot_route)))
        layer = SimpleNamespace(gate=lambda value: (ids, weights), shared_experts=mx.zeros_like)
        module = PagedExperts(layer, bank, hot, cold_fastpath=enabled)

        def project(base, meta, value, route, level, **kwargs):
            return mx.ones(value.shape, dtype=mx.float32)

        with patch('native_generate.projection_matvec_banked', side_effect=project) as projection:
            output = module(mx.ones((1, 1, 4), dtype=mx.float32))
            mx.eval(output)
        hot.route.assert_called_once()
        bank.select.assert_called_once_with((2, 5))
        bank.retain.assert_called_once()
        hot_bank.retain.assert_called_once()
        self.assertEqual(projection.call_count, 3)
        self.assertEqual(output.shape, (1, 1, 4))
        return module, projection.call_args_list

    def test_all_cold_fast_path_omits_refinement_inputs(self):
        module, calls = self.run_route(True, ())
        self.assertEqual(module._cold_only_calls, 1)
        self.assertEqual(module._cold_fastpath_calls, 1)
        for call in calls:
            self.assertIsNone(call.kwargs['refinements'])
            self.assertIsNone(call.kwargs['refined_route'])

    def test_reference_mode_keeps_mixed_inputs_for_all_cold_routes(self):
        module, calls = self.run_route(False, ())
        self.assertEqual(module._cold_only_calls, 1)
        self.assertEqual(module._cold_fastpath_calls, 0)
        for call in calls:
            self.assertIsNotNone(call.kwargs['refinements'])
            self.assertIsNotNone(call.kwargs['refined_route'])

    def test_one_ready_expert_preserves_mixed_execution(self):
        module, calls = self.run_route(True, (5,))
        self.assertEqual(module._cold_only_calls, 0)
        self.assertEqual(module._cold_fastpath_calls, 0)
        for call in calls:
            self.assertIsNotNone(call.kwargs['refinements'])
            self.assertIsNotNone(call.kwargs['refined_route'])

    def test_prefill_never_uses_the_decode_fast_path(self):
        module, calls = self.run_route(True, (), phase='prefill')
        self.assertEqual(module._cold_only_calls, 0)
        self.assertEqual(module._cold_fastpath_calls, 0)
        for call in calls:
            self.assertIsNotNone(call.kwargs['refinements'])
            self.assertIsNotNone(call.kwargs['refined_route'])


if __name__ == '__main__':
    unittest.main()
