import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import mlx.core as mx

from native_diagnostics import NativeDiagnostics
from native_generate import PagedExperts, generate, layer_major_prefill, main


class NativeDiagnosticsTest(unittest.TestCase):
    def test_flushed_records_survive_an_interrupted_phase(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'trace.jsonl'
            trace = NativeDiagnostics(path, True)
            try:
                trace.set_phase('decode')
                trace.begin_step(127)
                trace.record('checkpoint', bytes=123)
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                self.assertEqual(rows[-1]['decode_step'], 127)
                self.assertEqual(rows[-1]['bytes'], 123)
                with self.assertRaises(FileExistsError):
                    NativeDiagnostics(path)
                with self.assertRaises(ValueError):
                    trace.begin_step(128)
                with self.assertRaises(ValueError):
                    trace.record('invalid', seconds=float('nan'))
                self.assertEqual(len(path.read_text().splitlines()), 3)
            finally:
                trace.close()

    def run_routed(self, trace):
        planes = {name: ({}, {}) for name in ('gate', 'up', 'down')}
        bank = SimpleNamespace(fully_resident=False, layer=3, select=Mock(return_value=(0, 1)),
                               planes=planes, retain=Mock())
        hot_bank = SimpleNamespace(slots={5: 0}, planes={name: {} for name in planes}, retain=Mock())
        route = mx.array([0xffffffff, 0], dtype=mx.uint32)
        hot = SimpleNamespace(phase='decode', route=Mock(return_value=(hot_bank, route)))
        layer = SimpleNamespace(gate=lambda x: (mx.array([[[2, 5]]]), mx.array([[[.25, .75]]])),
                                shared_experts=lambda x: x * .5)
        module = PagedExperts(layer, bank, hot, diagnostics=trace)
        project = lambda base, meta, x, route, level, **kw: x + 1
        with patch('native_generate.projection_matvec_banked', side_effect=project), \
                patch('native_generate.mx.eval', wraps=mx.eval) as evaluate:
            output = module(mx.ones((1, 1, 4), dtype=mx.float32))
            calls = evaluate.call_count
        mx.eval(output)
        bank.select.assert_called_once_with((2, 5))
        bank.retain.assert_called_once_with(output)
        hot_bank.retain.assert_called_once_with(output)
        return output, calls

    def test_only_sync_mode_adds_a_layer_wait_and_preserves_arithmetic(self):
        reference, calls = self.run_routed(None)
        self.assertEqual(calls, 0)
        with tempfile.TemporaryDirectory() as directory:
            for sync in (False, True):
                with self.subTest(sync=sync):
                    path = Path(directory) / f'{sync}.jsonl'
                    trace = NativeDiagnostics(path, sync)
                    try:
                        trace.set_phase('decode')
                        trace.begin_step(0)
                        actual, calls = self.run_routed(trace)
                        self.assertEqual(calls, int(sync))
                        self.assertTrue(mx.array_equal(actual, reference).item())
                    finally:
                        trace.close()
                    rows = [json.loads(line) for line in path.read_text().splitlines()]
                    measured = [row for row in rows if row['event'] == 'routed_layer']
                    self.assertEqual(len(measured), int(sync))
                    if sync:
                        row = measured[0]
                        self.assertEqual(row['expert_ids'], [2, 5])
                        self.assertEqual(row['refined_slots'], [0xffffffff, 0])
                        intervals = [value for key, value in row.items()
                                     if key.endswith('_seconds') and key not in ('monotonic_seconds', 'call_wall_seconds')]
                        self.assertTrue(all(value >= 0 for value in intervals))
                        self.assertAlmostEqual(sum(intervals), row['call_wall_seconds'], places=7)

    def test_prefill_records_the_last_completed_and_started_layers_on_failure(self):
        class Layer:
            block_type = 'attention'

            def __call__(self, hidden, mask, state, indices):
                return hidden + 1, indices

        class FailingLayer(Layer):
            def __call__(self, *args):
                raise MemoryError('fixture pressure stop')

        embed = lambda ids: mx.ones((*ids.shape, 4))
        model = SimpleNamespace(model=SimpleNamespace(embed_tokens=embed), config=SimpleNamespace(hc_mult=1),
                                layers=[Layer(), FailingLayer()])
        cache = [SimpleNamespace(state=[]), SimpleNamespace(state=[])]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'trace.jsonl'
            trace = NativeDiagnostics(path)
            try:
                trace.set_phase('prefill')
                with patch('native_generate.memory_sample', return_value={'system_pressure': 1}), \
                        self.assertRaisesRegex(MemoryError, 'fixture pressure stop'):
                    layer_major_prefill(model, [[1, 2], [3]], cache, Mock(), None, trace)
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                phases = [(row['event'], row['layer']) for row in rows if 'layer' in row]
                self.assertEqual(phases, [('prefill_layer_started', 0), ('prefill_layer_complete', 0),
                                          ('prefill_layer_started', 1)])
            finally:
                trace.close()

    def test_cli_rejects_unsafe_diagnostic_combinations_before_loading(self):
        base = ['native_generate.py', '--model', '/missing', '--archive', '/missing', '--backbone', '/missing',
                '--output', '/missing/result.json']
        cases = [['--profile-sync'],
                 ['--diagnostics', '/missing/trace.jsonl', '--profile-sync', '--max-tokens', '129'],
                 ['--diagnostics', '/missing/trace.jsonl', '--token-major'],
                 ['--diagnostics', '/missing/result.json']]
        for arguments in cases:
            with self.subTest(arguments=arguments), patch('sys.argv', base + arguments), \
                    patch('native_generate.validate') as validate, patch('sys.stderr'), self.assertRaises(SystemExit) as error:
                main()
            self.assertEqual(error.exception.code, 2)
            validate.assert_not_called()

    def test_decode_intervals_and_immediate_eos_keep_the_existing_output_semantics(self):
        class Model:
            def make_cache(self):
                return []

            def __call__(self, *args, **kwargs):
                return SimpleNamespace(logits=mx.array([[[0., 1., 2.]]]))

        tokenizer = SimpleNamespace(decode=lambda ids, **kw: str(ids))
        with tempfile.TemporaryDirectory() as directory:
            for eos_immediately in (False, True):
                path = Path(directory) / f'{eos_immediately}.jsonl'
                trace = NativeDiagnostics(path, True)
                initial = mx.array([[[0., 1., 2.] if eos_immediately else [0., 2., 1.]]])
                try:
                    with patch('native_generate.layer_major_prefill', return_value=initial) as prefill, \
                            patch('native_generate.memory_sample', return_value={'system_pressure': 1}), \
                            patch('native_generate.stats', return_value={'count': 0}):
                        expected = generate(Model(), tokenizer, [1], 2, {2}, prefill, 1, None, Mock())
                        actual = generate(Model(), tokenizer, [1], 2, {2}, prefill, 1, None, Mock(), diagnostics=trace)
                finally:
                    trace.close()
                self.assertEqual(actual['generated_ids'], expected['generated_ids'])
                self.assertEqual(actual['text'], expected['text'])
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                steps = [row for row in rows if row['event'] == 'decode_step_complete']
                self.assertEqual(len(steps), int(not eos_immediately))
                if steps:
                    row = steps[0]
                    total = sum(row[key] for key in ('begin_policy_host_wall_seconds', 'forward_and_eval_wall_seconds',
                                                     'commit_host_wall_seconds', 'sampling_wall_seconds'))
                    self.assertAlmostEqual(total, row['timed_decode_wall_seconds'], places=7)
                else:
                    self.assertIsNone(actual['decode_tps'])


if __name__ == '__main__':
    unittest.main()
