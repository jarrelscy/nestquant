import pathlib,sys,unittest
import numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'spark/flash'))
from nq_flash_pool import Pool
from jmt_queuecost import QueueCostPlanner
from nq_flash_prefetch import ThroughputPrefetch
from test_flash_preload_cpu import Fake
class Tests(unittest.TestCase):
 def make(self,spares=1):
  pool=Pool([3],3,np.zeros((1,3),bool));pool.state[0,0]=2
  j=QueueCostPlanner(pool,[1],np.zeros((1,3),bool),np.ones((1,3)),clock=lambda:1.,risk=.5)
  j.latencies.append((1.,.1)) # arrival horizon2 at20TPS
  ex=Fake(pool,1+spares);ex.owned[3,0]=ex.free.pop()
  pf=ThroughputPrefetch(pool,ex,{3:1},lambda:j.priority,retention_priority=lambda:j.retention,paired_retention_priority=lambda:j.paired_retention,clock=lambda:1.)
  return pool,j,ex,pf
 def plan(self,pool,j,a):
  pool.wanted=j.plan(dict(activation=a,first_use=np.maximum.accumulate(a,axis=0)),None,1000,0,20)
 def test_prearrival_use_not_lost_during_paired_read(self):
  p,j,e,t=self.make();a=np.zeros((16,1,3));a[0,0,0]=1;a[2,0,1]=.8
  self.plan(p,j,a);self.assertEqual(j.lead,2);self.assertEqual(j.paired_retention[0,0],0);self.assertGreater(j.retention[0,0],1)
  t.pump(force=True);self.assertEqual(e.pre,[(3,1,0)]);self.assertEqual(p.state[0,0],2)
 def test_no_spare_cannot_ignore_early_eviction_loss(self):
  p,j,e,t=self.make(0);a=np.zeros((16,1,3));a[0,0,0]=1;a[2,0,1]=.8
  self.plan(p,j,a);t.pump(force=True);self.assertEqual(p.state[0,0],2);self.assertEqual(p.state[0,1],0)
 def test_future_reuse_protects_resident(self):
  p,j,e,t=self.make();a=np.zeros((16,1,3));a[2:,0,0]=.4;a[2:4,0,1]=1
  self.plan(p,j,a);self.assertTrue(p.wanted[0,0]);t.pump(force=True);self.assertFalse(e.pre)
 def test_ready_payload_gets_full_current_forecast_value(self):
  p,j,e,t=self.make();a=np.zeros((16,1,3));a[0,0,1]=1
  self.plan(p,j,a);self.assertEqual(j.priority[0,1],0);self.assertEqual(j.publication_priority[0,1],1)
 def test_reset_clears_transaction_costs(self):
  p,j,e,t=self.make();a=np.ones((16,1,3))*.1;self.plan(p,j,a);j.reset()
  for n in ['paired_retention','publication_priority','publication_retention']:self.assertFalse(getattr(j,n).any())
 def test_uniform_tie_prefers_resident(self):
  p,j,e,t=self.make();a=np.full((16,1,3),.2);self.plan(p,j,a);self.assertTrue(p.wanted[0,0]);t.pump(force=True);self.assertFalse(e.pre)
 def test_pending_pin_and_budget(self):
  p,j,e,t=self.make();p.state[0]=[0,1,0];a=np.zeros((16,1,3));a[:,:,2]=1
  self.plan(p,j,a);self.assertTrue(p.wanted[0,1]);self.assertEqual(p.wanted.sum(),1)
 def test_raises_on_bad_contract(self):
  p,j,e,t=self.make();a=np.full((16,1,3),2.)
  with self.assertRaises(ValueError):self.plan(p,j,a)
if __name__=='__main__':unittest.main()
