from contextlib import ExitStack, redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import native_generate


class NativeCliTest(unittest.TestCase):
    def arguments(self, *options):
        return ['native_generate.py', '--model', '/unused/model', '--archive', '/unused/archive',
                '--backbone', '/unused/backbone', '--output', '/unused/fresh-result.json', *options]

    def test_replay_requires_a_short_fixed_trace_without_other_diagnostics(self):
        for options in ([], ['--fixed-json', '/unused/trace.json', '--max-tokens', '129'],
                        ['--fixed-json', '/unused/trace.json', '--verify-prefill'],
                        ['--fixed-json', '/unused/trace.json', '--diagnostics', '/unused/trace.jsonl']):
            with self.subTest(options=options), patch('sys.argv', self.arguments('--replay-passes', '2', *options)), \
                    patch.object(Path, 'exists', return_value=False), patch.object(native_generate, 'validate') as load, \
                    redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as failure:
                    native_generate.main()
                self.assertEqual(failure.exception.code, 2)
                load.assert_not_called()

    def test_replay_with_a_short_fixed_trace_reaches_checkpoint_validation(self):
        options = ['--replay-passes', '2', '--fixed-json', '/unused/trace.json', '--max-tokens', '128']
        with patch('sys.argv', self.arguments(*options)), patch.object(Path, 'exists', return_value=False), \
                patch.object(native_generate, 'validate', side_effect=RuntimeError('validation sentinel')) as load:
            with self.assertRaisesRegex(RuntimeError, 'validation sentinel'):
                native_generate.main()
            load.assert_called_once_with(Path('/unused/model'))

    def test_replay_retains_model_and_first_pass_evidence_when_second_pass_fails(self):
        for fail_second in (False, True):
            with self.subTest(fail_second=fail_second), tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                output = Path(directory) / 'result.json'
                model = SimpleNamespace(parameters=lambda: {})
                bank = SimpleNamespace(bytes=16, staging=None, close=Mock())
                initial = {'generated_ids': [1, 2], 'decode': {'misses': 10}, 'decode_tps': 4.0}
                warm = {'generated_ids': [1, 2], 'decode': {'misses': 3}, 'decode_tps': 5.0}
                def read(path):
                    if path.name == 'config.json':
                        return json.dumps({'text_config': {'swiglu_limit': 10, 'vocab_size': 3}})
                    if path.name == 'generation_config.json':
                        return json.dumps({'eos_token_id': 0})
                    return json.dumps({'generated_ids': [1, 2]})
                args = self.arguments('--output', str(output), '--fixed-json', '/unused/trace.json',
                                      '--max-tokens', '2', '--replay-passes', '2')
                stack.enter_context(patch('sys.argv', args))
                stack.enter_context(patch.object(Path, 'read_text', read))
                for name in ('validate', 'MemoryGuard'):
                    stack.enter_context(patch.object(native_generate, name))
                stack.enter_context(patch.object(native_generate, 'stats', return_value={}))
                for name in ('set_cache_limit', 'set_memory_limit'):
                    stack.enter_context(patch.object(native_generate.mx, name))
                load = stack.enter_context(patch.object(native_generate, 'load_runtime', return_value=(model, [bank])))
                tokenizer = stack.enter_context(patch.object(native_generate.AutoTokenizer, 'from_pretrained'))
                tokenizer.return_value.apply_chat_template.return_value = [2]
                generate = stack.enter_context(patch.object(native_generate, 'generate', side_effect=[initial,
                    MemoryError('fixture pressure stop') if fail_second else warm]))
                stream = io.StringIO()
                stack.enter_context(redirect_stdout(stream))
                if fail_second:
                    with self.assertRaisesRegex(MemoryError, 'fixture pressure stop'):
                        native_generate.main()
                    self.assertFalse(output.exists())
                else:
                    native_generate.main()
                    with output.open() as handle:
                        result = json.load(handle)
                    self.assertEqual(result['decode']['misses'], 10)
                    self.assertEqual(result['warm_replay'], warm)
                load.assert_called_once()
                self.assertEqual(generate.call_count, 2)
                self.assertIs(generate.call_args_list[0].args[0], model)
                self.assertIs(generate.call_args_list[1].args[0], model)
                self.assertEqual(generate.call_args_list[0].args, generate.call_args_list[1].args)
                first = json.loads(stream.getvalue().splitlines()[0])
                self.assertEqual(first['pass_index'], 0)
                self.assertEqual(first['result']['decode']['misses'], 10)
                bank.close.assert_called_once()

    def test_invalid_decoder_loop_is_rejected_before_loading(self):
        with patch('sys.argv', self.arguments('--decoder-loop', 'unknown')), \
                patch.object(native_generate, 'validate') as load, redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as failure:
                native_generate.main()
            self.assertEqual(failure.exception.code, 2)
            load.assert_not_called()

    def test_five_percent_requires_refinements_and_six_gib_before_loading(self):
        for options in ([], ['--refinements'], ['--refinement-budget-gib', '6']):
            with self.subTest(options=options), patch('sys.argv', self.arguments('--hot-fraction', '0.05', *options)), \
                    patch.object(Path, 'exists', return_value=False), patch.object(native_generate, 'validate') as load, \
                    redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as failure:
                    native_generate.main()
                self.assertEqual(failure.exception.code, 2)
                load.assert_not_called()

    def test_staged_reads_reject_full_residency_before_loading(self):
        options = ['--base-read-mode', 'staged', '--full-base', '--rows', '288', '--base-budget-gib', '56']
        with patch('sys.argv', self.arguments(*options)), patch.object(Path, 'exists', return_value=False), \
                patch.object(native_generate, 'validate') as load, redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as failure:
                native_generate.main()
            self.assertEqual(failure.exception.code, 2)
            load.assert_not_called()

    def test_five_percent_with_six_gib_reaches_checkpoint_validation(self):
        options = ['--refinements', '--hot-fraction', '0.05', '--refinement-budget-gib', '6']
        with patch('sys.argv', self.arguments(*options)), patch.object(Path, 'exists', return_value=False), \
                patch.object(native_generate, 'validate', side_effect=RuntimeError('validation sentinel')) as load:
            with self.assertRaisesRegex(RuntimeError, 'validation sentinel'):
                native_generate.main()
            load.assert_called_once_with(Path('/unused/model'))


if __name__ == '__main__':
    unittest.main()
