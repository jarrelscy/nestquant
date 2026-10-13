import importlib.util,pathlib,tempfile,types,unittest,sys
from unittest.mock import patch
def fresh():
    spec=importlib.util.spec_from_file_location('sentinel',pathlib.Path(__file__).with_name('startup_sentinel.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
class Tests(unittest.TestCase):
    def fake(self):
        calls=[]
        def empty(size,**kw):
            calls.append(size)
            return types.SimpleNamespace(device=kw['device'],data_ptr=lambda:len(calls)*100)
        return types.SimpleNamespace(empty=empty,uint8='u8',device=lambda x:x,cuda=types.SimpleNamespace(synchronize=lambda:calls.append('sync'))),calls
    def test_reuse_and_exact_accounting_release(self):
        m=fresh();torch,calls=self.fake()
        with tempfile.TemporaryDirectory() as d,patch.dict(sys.modules,torch=torch):
            m.pathlib=types.SimpleNamespace(Path=lambda _:pathlib.Path(d))
            m.begin();a=m.acquire(m.EXPECTED_BYTES,'cuda:0');b=m.acquire(m.EXPECTED_BYTES,'cuda:0');self.assertIs(a,b)
            r=types.SimpleNamespace(model_memory_usage=m.EXPECTED_BYTES+123)
            m.mark_accounted(r);m.release();self.assertEqual(r.model_memory_usage,123)
            self.assertEqual(calls,[m.EXPECTED_BYTES,'sync']);self.assertIsNone(m._tensor)
            m.release();self.assertEqual(r.model_memory_usage,123)
    def test_outside_scope_original_allocations(self):
        m=fresh();torch,calls=self.fake()
        with patch.dict(sys.modules,torch=torch):
            self.assertIsNot(m.acquire(12,'cuda:0'),m.acquire(12,'cuda:0'))
        self.assertEqual(calls,[12,12])
    def test_fail_closed_shape_device_reentry(self):
        m=fresh();torch,_=self.fake()
        with patch.dict(sys.modules,torch=torch):
            m.begin()
            with self.assertRaises(RuntimeError):m.begin()
            with self.assertRaises(RuntimeError):m.acquire(1,'cuda:0')
            m.acquire(m.EXPECTED_BYTES,'cuda:0')
            with self.assertRaises(RuntimeError):m.acquire(m.EXPECTED_BYTES,'cuda:1')
    def test_exception_before_accounting_releases_without_subtraction(self):
        m=fresh();torch,_=self.fake()
        with tempfile.TemporaryDirectory() as d,patch.dict(sys.modules,torch=torch):
            m.pathlib=types.SimpleNamespace(Path=lambda _:pathlib.Path(d))
            try:
                m.begin();m.acquire(m.EXPECTED_BYTES,'cuda:0');raise ValueError('profile failure')
            except ValueError:pass
            finally:m.release()
            self.assertFalse(m._active);self.assertIsNone(m._tensor)
    def test_loader_order_and_profile_only_delta(self):
        p=pathlib.Path(__file__).parent;s=(p/'loader_memory_diagnostic.py').read_text()
        self.assertLess(s.index('before=torch.cuda.memory_allocated()'),s.index('startup_sentinel.begin()'))
        self.assertLess(s.index('startup_sentinel.begin()'),s.index('prewarm_mla.run(runner,x)'))
        self.assertLess(s.index('runner.model_memory_usage+='),s.index('startup_sentinel.mark_accounted(runner)'))
        self.assertIn('runtime._startup_runner=None\n            import startup_sentinel\n            startup_sentinel.release()',s)
        source=(p/'sparse_attn_indexer_kpool.py').read_text()
        self.assertEqual(source.count('startup_sentinel.acquire'),1)
        self.assertLess(source.index('if not isinstance(attn_metadata, dict):'),source.index('startup_sentinel.acquire'))
if __name__=='__main__':unittest.main()
