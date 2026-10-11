import pathlib,sys,unittest
import numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'spark/flash'))
from nq_flash_pool import Pool
from jmt_joint import JointPlanner
class JointTests(unittest.TestCase):
 def make(self):
  p=Pool([3],288,np.zeros((1,288),bool));p.state[0,:3]=2
  f=np.zeros_like(p.wanted);f[0,0]=True;clock=[0.]
  j=JointPlanner(p,[3],f,clock=lambda:clock[0],mix=1,width=4);p.observer=j
  return p,j,clock
 def test_measured_acknowledgment_delay_changes_lead(self):
  p,j,t=self.make();j.issued_ops([(3,4)],[]);p.state[0,4]=1
  t[0]=.2;p.landed(3,4)
  f=np.full((16,1,288),1/288);j.plan(f,np.zeros((1,288)),1000,0,30)
  self.assertEqual(j.lead,6);self.assertAlmostEqual(j.stats()['delivery_p90_ms'],200)
 def test_queue_backlog_extends_lead(self):
  p,j,t=self.make();f=np.full((16,1,288),1/288)
  j.plan(f,np.zeros((1,288)),100,29,30);self.assertEqual(j.lead,9)
 def test_joint_pool_protects_near_and_loads_future(self):
  p,j,t=self.make();p.state[0,2]=0
  f=np.zeros((16,1,288));f[:2,0,1]=1;f[2:,0,4]=1
  j.latencies.append((t[0],.1))
  wanted=j.plan(f,np.zeros((1,288)),1000,0,30)
  self.assertTrue(wanted[0,0] and wanted[0,1] and wanted[0,4]);self.assertEqual(wanted.sum(),3)
 def test_pending_and_fixed_survive(self):
  p,j,t=self.make();p.state[0,2]=0;p.state[0,9]=1
  f=np.zeros((16,1,288));f[:,:,4]=1
  wanted=j.plan(f,np.zeros((1,288)),1000,0,30)
  self.assertTrue(wanted[0,0] and wanted[0,9]);self.assertEqual(wanted.sum(),3)
 def test_regret_only_committed_first_cold_reuse(self):
  p,j,t=self.make();j.issued_ops([],[(3,1)]);p.state[0,1]=3;p.released(3,1)
  row=lambda pos:(pos,12,np.array([[1]]),np.array([[.5]]),np.array([2.]))
  j.committed([row(0)],np.zeros((1,1,1),bool));j.committed([row(1)],np.zeros((1,1,1),bool))
  self.assertEqual(j.regret_events,1);self.assertEqual(j.regret_mass,.5)
  with self.assertRaises(ValueError):j.committed([row(3)])
 def test_reset_ignores_old_request_callbacks(self):
  p,j,t=self.make();j.issued_ops([(3,4)],[(3,1)]);j.reset()
  j.event('landed',3,4);j.event('released',3,1)
  self.assertFalse(j.evicted);self.assertEqual(j.landed_count,0)
 def test_beyond_forecast_keeps_fixed_and_admits_useful_replacement(self):
  p,j,t=self.make();f=np.full((16,1,288),1/288)
  ema=np.zeros((1,288));ema[0,4]=1
  wanted=j.plan(f,ema,1,20,30)
  self.assertTrue(wanted[0,0] and wanted[0,4]);self.assertEqual(wanted.sum(),3)
  self.assertEqual(j.fallback_plans,1)

 def test_stale_latency_recovers_without_new_completions(self):
  p,j,t=self.make();j.latencies.append((0.,2.))
  f=np.full((16,1,288),1/288);ema=np.zeros((1,288));ema[0,4]=1
  j.plan(f,ema,1000,0,30);self.assertGreater(j.lead,16)
  t[0]=9.;j.plan(f,ema,1000,0,30)
  self.assertLessEqual(j.lead,16);self.assertEqual(j.plans,2)
 def test_request_reset_drops_latency_outliers(self):
  p,j,t=self.make();j.latencies.append((0.,2.));j.reset()
  self.assertEqual(len(j.latencies),0)

 def test_full_horizon_keeps_expert_needed_only_in_tail(self):
  p,j,t=self.make();f=np.zeros((16,1,288));f[:8,0,1]=.6;f[:8,0,4]=.4;f[8:,0,2]=1
  # With one-token delivery and an8-token window, resident2 is invisible.
  j.width=8;j.latencies.append((0.,.01))
  short=j.plan(f,np.zeros((1,288)),1000,0,30)
  self.assertFalse(short[0,2]);self.assertTrue(short[0,4])
  j.full_horizon=True;j.reload_penalty=.5
  full=j.plan(f,np.zeros((1,288)),1000,0,30)
  self.assertTrue(full[0,2]);self.assertFalse(full[0,4]);self.assertEqual(j.horizon_end,16)
 def test_full_horizon_can_replace_useless_resident(self):
  p,j,t=self.make();j.full_horizon=True;j.reload_penalty=.5
  f=np.zeros((16,1,288));f[:,:,4]=1
  wanted=j.plan(f,np.zeros((1,288)),1000,0,30)
  self.assertTrue(wanted[0,4]);self.assertTrue(wanted[0,0]);self.assertEqual(wanted.sum(),3)
 def test_last_horizon_counts_without_next_token_demand(self):
  p,j,t=self.make();j.full_horizon=True;j.reload_penalty=0
  f=np.zeros((16,1,288));f[:15,0,1]=1;f[15,0,2]=1
  wanted=j.plan(f,np.zeros((1,288)),1000,0,30)
  self.assertTrue(wanted[0,2]);self.assertEqual(wanted.sum(),3)
