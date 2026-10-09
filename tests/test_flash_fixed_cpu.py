import os,pathlib,sys,unittest
import numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'spark/flash'))
from fixed_policy import FixedSetPolicy
from jt_runtime import module, layer_budgets, fixed_counts_for_fraction
from nq_flash_pool import Pool

class Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  root=pathlib.Path(os.environ.get('NQ_FLASH_TEST_MODEL','/tmp/nestquant/flash-fp8-backbone'))
  cls.Policy=module(root/'serving/predictor/jt/policy.py','fixed_test_reference').FloatingSet
 def test_ten_percent_per_layer(self):
  b=layer_budgets({},'spark_256K_U_2630_flat50');c=fixed_counts_for_fraction(b,'0.1')
  self.assertEqual(sum(c.values()),263)
  self.assertEqual(sum(b.values())-sum(c.values()),2367)
  self.assertTrue(all(abs(c[L]-b[L]/10)<1 for L in b))
  self.assertEqual(c,fixed_counts_for_fraction(dict(reversed(list(b.items()))),'0.1'))
  for bad in [-1,1,2]:
   with self.assertRaises(ValueError):fixed_counts_for_fraction(b,bad)
 def test_zero_fixed_exact_reference(self):
  a=self.Policy(48,np.arange(48));b=FixedSetPolicy(self.Policy,48,np.arange(48),[]);rng=np.random.default_rng(7)
  for t in range(100):
   block=rng.random(288);block/=block.sum()
   x,loads=a.before_row(block);y,other=b.before_row(block)
   np.testing.assert_array_equal(x,y);self.assertEqual(loads,other)
   ids=rng.integers(0,288,8);w=rng.random(8);a.after_row(ids,w,3);b.after_row(ids,w,3)
   np.testing.assert_array_equal(a.state,b.state)
 def test_fixed_survive_zero_score_and_never_demote(self):
  fixed=np.array([2,7,15,31,63,127,200,287]);default=np.r_[fixed,[i for i in range(288) if i not in fixed]][:48]
  p=FixedSetPolicy(self.Policy,48,default,fixed);pool=Pool([3],288,p.cur[None,:]);pool.state[pool.wanted]=2
  rng=np.random.default_rng(17)
  for t in range(160):
   block=rng.random(288);block[fixed]=0;block/=block.sum()
   before=p.cur.copy();wanted,loads=p.before_row(block)
   self.assertTrue(wanted[fixed].all());self.assertEqual(wanted.sum(),48)
   if t%8:self.assertTrue(np.array_equal(before,wanted))
   pool.wanted=wanted[None,:];up,down=pool.operations()
   self.assertFalse(any(e in fixed for _,e in down))
   for L,e in down:pool.released(L,e)
   for L,e in up:pool.landed(L,e)
   ids=np.array([i for i in range(288) if i not in fixed])[rng.integers(0,280,8)];p.after_row(ids,rng.random(8),2)
  reset=FixedSetPolicy(self.Policy,48,default,fixed)
  self.assertTrue(reset.cur[fixed].all());self.assertEqual(reset.t,0);self.assertEqual(reset.state.sum(),0)
 def test_invalid_fixed_configuration(self):
  for fixed,total in [([1,1],48),([-1],48),([288],48),(list(range(48)),48),([287],48)]:
   with self.assertRaises(ValueError):FixedSetPolicy(self.Policy,total,np.arange(48),fixed)

if __name__=='__main__':unittest.main()
