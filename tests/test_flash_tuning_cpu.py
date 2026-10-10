import pathlib,sys,unittest,json
import numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'spark/flash'))
from jt_runtime import policy_parameters,CommittedPredictor,module
from fixed_policy import FixedSetPolicy
from functools import partial
class TuningTests(unittest.TestCase):
 def test_bounds(self):
  for d in [{'mix':1.1},{'G':0},{'G':8.0},{'hm':-1},{'half_life':0},{'unknown':1},{'mix':float('nan')},{'hm':True}]:
   with self.assertRaises(ValueError):policy_parameters(json.dumps(d))
 def test_priority_matches_selection_mix(self):
  p=CommittedPredictor.__new__(CommittedPredictor)
  class Policy:state=np.array([3.,1.])
  p.policies=[Policy()];p.block=np.array([[.1,.9]])
  for mix in (0,.5,1):
   p.parameters=policy_parameters(json.dumps({'mix':mix}))
   np.testing.assert_allclose(p.loading_priority(),(1-mix)*np.array([[.75,.25]])+mix*p.block)
 def test_reset_preserves_parameters_and_fixed(self):
  ref=module('/data/models/jarrelscy/GLM-5.3-Flash-NestQuant-1.5-4bit/serving/predictor/jt/policy.py','policy_test_tune').FloatingSet
  p=CommittedPredictor.__new__(CommittedPredictor)
  class Net:
   def reset(self):pass
  p.net=Net();p.layers=[3];p.budgets={3:10};p.defaults=[np.arange(10)];p.fixed_ids=[np.array([0,1])];p.n_fixed=2
  p.parameters=policy_parameters('{"mix":1,"hm":1,"half_life":32}')
  p.policy_cls=partial(ref,**p.parameters)
  for _ in range(2):
   p.reset();a=p.policies[0]
   self.assertEqual(a.mix,1);self.assertEqual(a.hm,1);self.assertEqual(a.a,.5**(1/32));self.assertEqual(a.cur.sum(),10)
   for t in range(20):
    b=np.zeros(288);b[20:30]=.1;a.before_row(b);a.after_row(np.arange(8),np.ones(8),1)
    self.assertTrue(a.cur[0] and a.cur[1]);self.assertEqual(a.cur.sum(),10)
 def test_committed_batches_refresh_one_and_two(self):
  rng=np.random.default_rng(612)
  blocks=rng.random((12,42,288));blocks/=blocks.sum(-1,keepdims=True)
  ref=module('/data/models/jarrelscy/GLM-5.3-Flash-NestQuant-1.5-4bit/serving/predictor/jt/policy.py','policy_batch_tune').FloatingSet
  for G in (1,2,4):
   class Net:
    def reset(self):self.t=0
    def predict(self,tok,ids,q):
     b=blocks[self.t:self.t+len(tok)];self.t+=len(tok);return b
   p=CommittedPredictor.__new__(CommittedPredictor);p.net=Net();p.layers=list(range(3,45));p.budgets={L:10 for L in p.layers};p.defaults=[np.arange(10) for L in p.layers]
   p.parameters=policy_parameters(json.dumps(dict(mix=.75,G=G)))
   p.policy_cls=partial(ref,**p.parameters);p.reset()
   oracle=[p.policy_cls(10,np.arange(10)) for _ in p.layers]
   rows=[(t,t,np.stack([rng.choice(288,8,False) for _ in p.layers]),rng.random((42,8)),rng.random(42)) for t in range(12)]
   for start in range(0,12,3):
    p.commit(rows[start:start+3])
    for row in rows[start:start+3]:
     t,_,ids,w,xn=row
     for i,o in enumerate(oracle):o.after_row(ids[i],w[i],xn[i]);o.before_row(blocks[t,i])
    np.testing.assert_array_equal(p.prepare(),np.stack([o.cur for o in oracle]))
   p.reset();self.assertEqual(p.t,0);self.assertIsNone(p.block)
if __name__=='__main__':unittest.main()
