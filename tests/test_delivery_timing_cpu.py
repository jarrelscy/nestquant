import unittest,sys,pathlib,types
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'spark/flash'))
from delivery_timing import PhaseMeter
class T(unittest.TestCase):
 def test_nested_exclusive_and_midphase_snapshot(self):
  now=[0.];m=PhaseMeter(lambda:now[0]);o=types.SimpleNamespace()
  def inner():now[0]+=3
  o.inner=inner;m.wrap(o,'inner','inner')
  def outer():
   now[0]+=2;s=m.snapshot();o.inner();now[0]+=4;return m.delta(s)
  o.outer=outer;m.wrap(o,'outer','outer');d=o.outer()
  self.assertEqual(d['outer'],4);self.assertEqual(d['inner'],3)
  self.assertEqual(m.total['outer'],6);self.assertEqual(m.total['inner'],3)
 def test_exception_unwinds(self):
  m=PhaseMeter();o=types.SimpleNamespace(f=lambda:(_ for _ in ()).throw(ValueError()))
  m.wrap(o,'f','fail')
  with self.assertRaises(ValueError):o.f()
  self.assertEqual(m.stack,[])
if __name__=='__main__':unittest.main()
