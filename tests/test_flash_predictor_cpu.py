"""Synthetic causal, cache, rejection and policy tests; no calibration traces."""
import os,pathlib,sys,unittest
import numpy as np
import torch
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'spark/flash'))
from jt_runtime import IncrementalJT,RowLedger,module,CommittedPredictor,layer_budgets

RELEASE=os.environ.get('NQ_FLASH_TEST_MODEL')

class LedgerTests(unittest.TestCase):
 def test_reject_and_accept(self):
  l=RowLedger();ids=np.zeros((2,42,8),np.int64);w=np.ones_like(ids,dtype=np.float32);xn=np.ones((2,42))
  l.stage([0,1],[11,999],ids,w,xn)
  self.assertEqual([r[1] for r in l.accept_before(1)],[11])
  l.stage([1,2],[22,33],ids,w,xn)
  self.assertEqual([r[1] for r in l.accept_before(3)],[22,33])
  self.assertEqual(l.committed_end,3)
  l.reset();self.assertEqual(l.committed_end,0)
 def test_two_drafts_all_rejection_boundaries(self):
  for rejected in (0,1,2):
   l=RowLedger();ids=np.zeros((3,42,8),np.int64);w=np.ones_like(ids,dtype=np.float32);xn=np.ones((3,42))
   l.stage([0,1,2],[10,20,30],ids,w,xn)
   count=3-rejected
   self.assertEqual([r[1] for r in l.accept_before(count)],[10,20,30][:count])
   # A corrected token replaces the rejected suffix on the next target step.
   l.stage([count],[99],ids[:1],w[:1],xn[:1])
   self.assertEqual([r[1] for r in l.accept_before(count+1)],[99])
   l.reset();self.assertEqual(l.committed_end,0);self.assertFalse(l.pending)
 def test_missing_prefix_refused(self):
  l=RowLedger()
  with self.assertRaises(ValueError):l.stage([20],[1],[[]],[[]],[[]])

@unittest.skipUnless(RELEASE,'Set NQ_FLASH_TEST_MODEL to the public checkpoint')
class PredictorTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  p=pathlib.Path(RELEASE)/'serving/predictor/jt'
  cls.ref=module(p/'jt_model.py','release_jt_test')
  cls.policy=module(p/'policy.py','release_policy_test').FloatingSet
 def test_cached_inference_matches_full_causal_reference(self):
  torch.manual_seed(12)
  m=self.ref.JT(d=32,nl=2,nh=4,dt=8,win=8).eval()
  tok=torch.arange(27)[None];ids=torch.randint(0,288,(1,27,42,8));q=torch.rand(1,27,42,8);q/=q.sum(-1,keepdim=True)
  with torch.inference_mode():expected=m(tok,ids,q)[1][0].numpy()
  for stride in (1,3,9):
   c=IncrementalJT(m,self.ref.rope)
   out=np.concatenate([c.predict(tok[0,i:i+stride],ids[0,i:i+stride],q[0,i:i+stride]) for i in range(0,27,stride)])
   np.testing.assert_allclose(out,expected,rtol=3e-5,atol=1e-7)
   self.assertLessEqual(c.cache[0][0].shape[-2],7)
   c.reset();self.assertEqual(c.position,0)
 def test_u_distribution_layer_budget_policy_parity(self):
  self.u_distribution=True
  self.test_exact_policy_and_causal_inputs()
 def test_u2630_policy_parity(self):
  self.u_distribution=True;self.u_preset='spark_256K_U_2630';self.u_total=2630
  b=layer_budgets({},self.u_preset);old=layer_budgets({},'spark_256K_U_2504')
  self.assertEqual(sum(b.values()),2630)
  self.assertTrue(all(b[L]>=old[L] for L in old))
  self.test_exact_policy_and_causal_inputs()
 def test_u2630_flat50_policy_parity(self):
  self.u_distribution=True;self.u_preset='spark_256K_U_2630_flat50';self.u_total=2630
  b=layer_budgets({},self.u_preset)
  self.assertEqual(sum(b.values()),2630)
  self.assertEqual((min(b.values()),max(b.values())),(48,114))
  self.assertEqual(set(b),set(range(3,45)))
  self.test_exact_policy_and_causal_inputs()
 def test_layer_presets_preserve_total_and_source_order(self):
  b=layer_budgets({},'spark_128K_U_2352')
  self.assertEqual(sum(b.values()),2352)
  self.assertEqual(b[3],147);self.assertEqual(b[35],29)
  self.assertGreater(b[42],b[28]);self.assertEqual(set(b),set(range(3,45)))
  legacy={'n_float':{'spark_128K':{'3-17':102,'18-44':74}}}
  self.assertEqual(sum(layer_budgets(legacy,'spark_128K').values()),3528)
  self.assertEqual(sum(layer_budgets({},'spark_128K_74_46').values()),2352)
 def test_exact_policy_and_causal_inputs(self):
  rng=np.random.default_rng(192)
  blocks=rng.random((40,42,288));blocks/=blocks.sum(-1,keepdims=True)
  class Net:
   def __init__(s):s.position=0;s.inputs=[]
   def reset(s):s.position=0;s.inputs=[]
   def predict(s,tok,ids,q):
    s.inputs.extend(zip(tok,ids.copy(),q.copy()));i=s.position;s.position+=len(tok);return blocks[i:s.position]
  p=CommittedPredictor.__new__(CommittedPredictor)
  p.net=Net();p.layers=list(range(3,45));p.budgets=(layer_budgets({},getattr(self,'u_preset','spark_128K_U_2352')) if getattr(self,'u_distribution',False) else {L:102 if L<18 else 74 for L in p.layers})
  p.defaults=[np.arange(p.budgets[L]) for L in p.layers];p.policy_cls=self.policy;p.reset()
  oracle=[self.policy(p.budgets[L],d) for L,d in zip(p.layers,p.defaults)]
  rows=[(t,100+t,rng.integers(0,288,(42,8)),rng.random((42,8)),rng.random(42)) for t in range(40)]
  for t,row in enumerate(rows):
   for i,o in enumerate(oracle):o.before_row(None if t==0 else blocks[t-1,i]);o.after_row(row[2][i],row[3][i],row[4][i])
   want=p.commit([row])
   for i,o in enumerate(oracle):
    o.before_row(blocks[t,i])
    np.testing.assert_array_equal(want[i],o.cur)
    np.testing.assert_array_equal(p.policies[i].state,o.state)
   # The next row was prepared already; skip duplicate before_row in oracle.
   for o in oracle:
    original=o.before_row
    if t<39:
     def once(block,_o=o,_orig=original):
      _o.before_row=_orig;return _o.cur.copy(),[]
     o.before_row=once
  self.assertEqual(int(want.sum()),getattr(self,'u_total',2352) if getattr(self,'u_distribution',False) else 3528)
  for t,(token,ids,q) in enumerate(p.net.inputs):
   self.assertEqual(token,100+t)
   if t:np.testing.assert_array_equal(ids,rows[t-1][2])
   else:self.assertEqual(q.sum(),0)
  p.reset();self.assertEqual(p.t,0);self.assertIsNone(p.block)

if __name__=='__main__':unittest.main()
