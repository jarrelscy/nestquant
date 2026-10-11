import sys,pathlib,unittest
import numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'spark/flash'))
from nq_flash_pool import Pool
from jmt_probability_planner import ProbabilityPlanner
class Tests(unittest.TestCase):
 def setup(self):
  pool=Pool([3],288,np.zeros((1,288),bool));pool.state[0,0]=2
  fixed=np.zeros((1,288),bool)
  j=ProbabilityPlanner(pool,[1],fixed,np.ones((1,288)),clock=lambda:0)
  return pool,j
 def f(self,p):return {'activation':p,'first_use':np.maximum.accumulate(p,axis=0)}
 def test_probability_not_softmax(self):
  pool,j=self.setup();p=np.full((16,1,288),.02);p[:,:,1]=.8
  w=j.plan(self.f(p),None,1000,0,30);self.assertTrue(w[0,1]);self.assertEqual(w.sum(),1)
 def test_benefit_can_outweigh_frequency(self):
  pool,j=self.setup();p=np.zeros((16,1,288));p[:,:,1]=.8;p[:,:,2]=.2;j.benefit[0,2]=10
  self.assertTrue(j.plan(self.f(p),None,1000,0,30)[0,2])
 def test_delay_excludes_unreachable_value(self):
  pool,j=self.setup();p=np.zeros((16,1,288));p[:2,0,1]=1;p[3:,0,2]=.5
  j.latencies.append((0,.1));w=j.plan(self.f(p),None,1000,0,30)
  self.assertTrue(w[0,2]);self.assertEqual(j.lead,3)
 def test_cdf_risk_protects_resident(self):
  pool,j=self.setup();p=np.zeros((16,1,288));p[:,0,0]=.1;p[1:,0,1]=1.7/15
  f=self.f(p);f['first_use'][1:,0,0]=.95;j.latencies.append((0,.1))
  # Resident total1.6 vs incoming1.5867; increase replacement slightly.
  p[2:,0,1]=1.8/14;f=self.f(p);f['first_use'][1:,0,0]=.95
  j.reload_penalty=0;self.assertTrue(j.plan(f,None,1000,0,30)[0,1])
  j.reload_penalty=.5;self.assertTrue(j.plan(f,None,1000,0,30)[0,0])
 def test_fixed_and_pending_pinned(self):
  pool,j=self.setup();pool.state[0,0]=1;p=np.zeros((16,1,288));p[:,:,2]=1
  self.assertTrue(j.plan(self.f(p),None,1000,0,30)[0,0])
 def test_fallback_does_not_use_ema(self):
  pool,j=self.setup();p=np.zeros((16,1,288));p[:,:,2]=1;ema=np.zeros((1,288));ema[0,3]=1e10
  self.assertTrue(j.plan(self.f(p),ema,1,50,30)[0,2]);self.assertEqual(j.fallback_plans,1)
 def test_invalid_cdf_rejected(self):
  pool,j=self.setup();p=np.full((16,1,288),.1);f=self.f(p);f['first_use'][3]=0
  with self.assertRaises(ValueError):j.plan(f,None,1000,0,30)
if __name__=='__main__':unittest.main()
