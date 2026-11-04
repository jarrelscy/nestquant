import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import benchmark
from verify_generation import answer_matches


class AnswerValidationTests(unittest.TestCase):
    def test_exact_text(self):
        self.assertTrue(answer_matches(' GLM_MAC_OK\n', 'GLM_MAC_OK'))
        self.assertFalse(answer_matches('GLM_MAC_OK extra text', 'GLM_MAC_OK'))

    def test_integer_array(self):
        self.assertTrue(answer_matches('[1, 2, 3]', [1, 2, 3]))
        for response in ('[true, 2, 3]', '[1.0, 2, 3]', '["1", 2, 3]', '[3, 2, 1]',
                         '[1, 2]', '[1, 2, 3, 4]', '{"1": 2}', 'null', '```json\n[1, 2, 3]\n```'):
            with self.subTest(response=response):
                self.assertFalse(answer_matches(response, [1, 2, 3]))


class BenchmarkValidationTests(unittest.TestCase):
    def run_benchmark(self, tokens):
        report = dict(decode_step_seconds=[0.2] * max(0, tokens - 1), generated_tokens=tokens, generated_ids=list(range(tokens)), decode_tokens_per_second=5.0,
                      ttft_seconds=1.0, prefill_tokens_per_second=10.0, disk_reads=dict(prefill=0, decode=0),
                      resources=dict(after_decode=dict(lifetime_peak_physical_footprint_bytes=1024)))
        runtime = MagicMock(hot_slots=2630)
        runtime.store.base_bytes = 0
        runtime.generate.return_value = report
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'results'
            argv = ['benchmark.py', '--model', directory, '--output-dir', str(output), '--tokens', '16']
            with patch('sys.argv', argv), patch.object(benchmark, 'Runtime', return_value=runtime), \
                 patch.object(benchmark.mx, 'device_info', return_value={'device': 'fixture'}), \
                 patch.object(benchmark.mx, 'set_cache_limit'), patch.object(benchmark.torch, 'set_num_threads'), \
                 contextlib.redirect_stdout(io.StringIO()):
                if tokens < 16:
                    with self.assertRaisesRegex(RuntimeError, '16 required'):
                        benchmark.main()
                    self.assertTrue((output / 'run-1.json').exists())
                    self.assertFalse((output / 'summary.json').exists())
                else:
                    benchmark.main()
                    summary = json.loads((output / 'summary.json').read_text())
                    self.assertEqual(summary['status'], 'complete')
                    self.assertEqual(summary['generated_tokens'], [16, 16, 16])
            runtime.close.assert_called_once()

    def test_early_eos_does_not_produce_throughput_summary(self):
        self.run_benchmark(2)

    def test_full_length_runs_produce_summary(self):
        self.run_benchmark(16)


if __name__ == '__main__':
    unittest.main()
