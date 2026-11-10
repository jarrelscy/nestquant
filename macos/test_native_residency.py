import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from native_generate import main


class NativeResidencyTest(unittest.TestCase):
    def test_paged_limit_is_optional_and_applied_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'config.json').write_text(json.dumps({'text_config': {'swiglu_limit': 10}}))
            (root / 'generation_config.json').write_text(json.dumps({'eos_token_id': 2}))
            base = ['native_generate.py', '--model', directory, '--archive', directory,
                    '--backbone', directory, '--output', str(root / 'result.json')]
            for limit in (None, 0, 32):
                arguments = [] if limit is None else ['--paged-wired-gib', str(limit)]
                with self.subTest(limit=limit), patch('sys.argv', base + arguments), \
                        patch('native_generate.validate'), patch('native_generate.mx.set_cache_limit'), \
                        patch('native_generate.mx.set_memory_limit'), patch('native_generate.MemoryGuard'), \
                        patch('native_generate.mx.set_wired_limit') as wired:
                    def stop_at_load(*args):
                        if limit is None:
                            wired.assert_not_called()
                        else:
                            wired.assert_called_once_with(limit * 1024**3)
                        raise RuntimeError('fixture load boundary')

                    with patch('native_generate.load_runtime', side_effect=stop_at_load), \
                            self.assertRaisesRegex(RuntimeError, 'fixture load boundary'):
                        main()

    def test_invalid_paged_limits_fail_before_checkpoint_validation(self):
        base = ['native_generate.py', '--model', '/missing', '--archive', '/missing', '--backbone', '/missing',
                '--output', '/missing/result.json']
        cases = [['--paged-wired-gib', '-1'], ['--paged-wired-gib', '64'],
                 ['--paged-wired-gib', '0', '--full-base', '--rows', '288', '--base-budget-gib', '56'],
                 ['--paged-wired-gib', '32', '--full-base', '--rows', '288', '--base-budget-gib', '56']]
        for arguments in cases:
            with self.subTest(arguments=arguments), patch('sys.argv', base + arguments), \
                    patch('native_generate.validate') as validate, patch('native_generate.mx.set_wired_limit') as wired, \
                    patch('sys.stderr'), self.assertRaises(SystemExit) as error:
                main()
            self.assertEqual(error.exception.code, 2)
            validate.assert_not_called()
            wired.assert_not_called()


if __name__ == '__main__':
    unittest.main()
