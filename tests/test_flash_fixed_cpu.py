import os,pathlib,sys,unittest
import numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'spark/flash'))
from fixed_policy import FixedSetPolicy
from jt_runtime import module, layer_budgets, fixed_counts_for_fraction, initial_expert_pools
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
 def test_salience_not_frequency_and_zero_unchanged(self):
  import copy
  counts=np.arange(288,dtype=float);scores=np.arange(288,0,-1,dtype=float)
  manifest={'n_routed':{'3':counts.tolist()},'score':{'3':scores.tolist()},'fixed_set':{'3':list(range(19))}}
  default,fixed=initial_expert_pools(manifest,[3],{3:50},{3:10})
  np.testing.assert_array_equal(fixed[0],np.arange(10))
  np.testing.assert_array_equal(default[0][10:],np.arange(287,247,-1))
  self.assertEqual(len(np.unique(default[0])),50)
  nofixed,_=initial_expert_pools({'n_routed':manifest['n_routed']},[3],{3:50},{3:0})
  np.testing.assert_array_equal(nofixed[0],np.argsort(-counts,kind='stable')[:50])
  bad=copy.deepcopy(manifest);bad['score']['3'][0]=float('nan')
  with self.assertRaises(ValueError):initial_expert_pools(bad,[3],{3:50},{3:10})
  bad=copy.deepcopy(manifest);bad['fixed_set']['3'][0]=287
  with self.assertRaises(ValueError):initial_expert_pools(bad,[3],{3:50},{3:10})
 def test_all_shipped_layers_and_fixed_policy(self):
  import json
  root=pathlib.Path(os.environ.get('NQ_FLASH_TEST_MODEL','/tmp/nestquant/flash-fp8-backbone'))
  manifest=json.loads((root/'fixed_set.json').read_text())
  budgets=layer_budgets({},'spark_256K_U_2630_flat50');counts=fixed_counts_for_fraction(budgets,.2)
  self.assertEqual(sum(counts.values()),526)
  defaults,fixed=initial_expert_pools(manifest,list(budgets),budgets,counts)
  for L,d,f in zip(budgets,defaults,fixed):
   ranked=sorted(range(288),key=lambda e:(-manifest['score'][str(L)][e],e))
   self.assertEqual(f.tolist(),ranked[:counts[L]])
   policy=FixedSetPolicy(self.Policy,budgets[L],d,f)
   for t in range(17):
    current,_=policy.before_row(np.ones(288)/288)
    self.assertTrue(current[f].all());self.assertEqual(current.sum(),budgets[L])
    policy.after_row(np.arange(8),np.ones(8)/8,1.)
 def test_invalid_fixed_configuration(self):
  for fixed,total in [([1,1],48),([-1],48),([288],48),(list(range(48)),48),([287],48)]:
   with self.assertRaises(ValueError):FixedSetPolicy(self.Policy,total,np.arange(48),fixed)

if __name__=='__main__':unittest.main()
