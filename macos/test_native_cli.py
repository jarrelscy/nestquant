from contextlib import redirect_stderr
import io
from pathlib import Path
import unittest
from unittest.mock import patch

import native_generate


class NativeCliTest(unittest.TestCase):
    def arguments(self, *options):
        return ['native_generate.py', '--model', '/unused/model', '--archive', '/unused/archive',
                '--backbone', '/unused/backbone', '--output', '/unused/fresh-result.json', *options]

    def test_five_percent_requires_refinements_and_six_gib_before_loading(self):
        for options in ([], ['--refinements'], ['--refinement-budget-gib', '6']):
            with self.subTest(options=options), patch('sys.argv', self.arguments('--hot-fraction', '0.05', *options)), \
                    patch.object(Path, 'exists', return_value=False), patch.object(native_generate, 'validate') as load, \
                    redirect_stderr(io.StringIO()):
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
