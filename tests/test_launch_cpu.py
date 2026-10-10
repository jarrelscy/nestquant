"""CPU-only launcher defaults, override and second-drive recovery checks."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('prepare_dual', ROOT / 'sm120/serve/tools/prepare_dual.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

class LauncherTests(unittest.TestCase):
    def test_dual_copy_and_interrupted_refresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / 'src', Path(tmp) / 'dst'
            src.mkdir()
            for r in range(4):
                (src / f'rank{r}.bin').write_bytes(bytes([r]) * 100)
                (src / f'rank{r}.json').write_text('{}')
            (src / 'artifact_stamp.json').write_text('{"version":1}')
            mod.prepare(src, dst)
            self.assertEqual((dst / 'rank3.bin').read_bytes(), bytes([3]) * 100)
            before = (dst / 'rank3.bin').stat().st_mtime_ns
            mod.prepare(src, dst)
            self.assertEqual((dst / 'rank3.bin').stat().st_mtime_ns, before)
            (dst / '.nq-copy-incomplete').touch()
            (dst / 'rank3.bin').write_bytes(b'bad')
            mod.prepare(src, dst)
            self.assertEqual((dst / 'rank3.bin').read_bytes(), bytes([3]) * 100)
            self.assertFalse((dst / '.nq-copy-incomplete').exists())
            with self.assertRaises(ValueError):
                mod.prepare(src, src)

    def config(self, **overrides):
        env = {k: v for k, v in os.environ.items() if not k.startswith(('NQ_', 'COMPOSE_', 'NUM_SPEC', 'PARALLEL'))}
        env.update(VLLM_API_KEY='SECRET_SENTINEL_NOT_FOR_OUTPUT', **overrides)
        result = subprocess.check_output(['bash', str(ROOT / 'start.sh'), 'config'], env=env, text=True)
        self.assertNotIn('SECRET_SENTINEL_NOT_FOR_OUTPUT', result)
        return result

    def test_defaults_and_optimizations(self):
        out = self.config()
        for value in ['NQ_JOINT_FIXED=1', 'NQ_SLOTS_PER_LAYER=56', 'NQ_SCHED=tap',
                      'NQ_LMPF=1', 'NQ_LMPF_CG=1', 'NQ_LMPF_BORROW=all',
                      'NQ_PREFILL_BORROW=1', 'NQ_PREFILL_KV_OFFLOAD=1',
                      'NQ_TAP_TODO_FIX=2', 'NQ_DEC_ASYNC=3', 'NQ_FOLLOW_COALESCE=1',
                      'ENABLE_LMCACHE=1', 'NUM_SPEC=3', 'MAXLEN=1048576', 'MAX_NUM_SEQS=1']:
            self.assertIn(value, out)

    def test_overrides_preserved(self):
        out = self.config(NQ_JOINT_FIXED='0', NQ_SLOTS_PER_LAYER='80', NQ_LMPF='0')
        for value in ['NQ_JOINT_FIXED=0', 'NQ_SLOTS_PER_LAYER=80', 'NQ_LMPF=0', 'NOTE:']:
            self.assertIn(value, out)

if __name__ == '__main__':
    unittest.main()
