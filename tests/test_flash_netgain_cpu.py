import pathlib,sys,unittest,types,importlib.util
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'spark/flash'))
from nq_flash_pool import Pool
from nq_flash_prefetch import ThroughputPrefetch
from jmt_probability_planner import ProbabilityPlanner
from test_flash_preload_cpu import Fake
class Tests(unittest.TestCase):
 def make(self,spares=1):
  p=Pool([3,4],2,np.zeros((2,2),bool));p.state[:,0]=2;p.wanted[:,1]=True
  e=Fake(p,2+spares)
  for L in p.layers:e.owned[L,0]=e.free.pop()
  incoming=np.array([[.9,1.],[0.,.8]])
  retention=incoming.copy()
  t=ThroughputPrefetch(p,e,{3:1,4:1},lambda:incoming,retention_priority=lambda:retention,max_pending=2,min_pending=1,clock=lambda:0.)
  return p,e,t,incoming,retention
 def test_netgain_spare(self):
  p,e,t,_,_=self.make();t.pump(force=True);self.assertEqual(e.pre,[(4,1,0)])
 def test_netgain_without_spare(self):
  p,e,t,_,_=self.make(0);t.pump(force=True);self.assertEqual(p.state[:,0].tolist(),[2,3])
 def test_negative_gain_never_evicts(self):
  p,e,t,_,r=self.make();r[:,0]=10;t.pump(force=True);self.assertFalse(e.pre);self.assertTrue(np.all(p.state[:,0]==2))
 def test_empty_capacity_has_no_victim_cost(self):
  p,e,t,_,r=self.make();p.state[0,0]=0;e.free.append(e.owned.pop((3,0)));r[:,0]=10
  t.pump(force=True);self.assertEqual(p.state[0,1],1);self.assertEqual(p.state[1,0],2)
 def test_planner_preserves_and_resets_retention(self):
  p=Pool([3],4,np.zeros((1,4),bool));p.state[0,:2]=2
  j=ProbabilityPlanner(p,[2],p.wanted.copy(),np.ones((1,4)),clock=lambda:1.,risk=.5);j.latencies.append((1.,.15))
  a=np.zeros((16,1,4));a[:2,0,0]=1;a[2:,0,1]=.1;a[2:,0,2:]=.2
  p.wanted=j.plan(dict(activation=a,first_use=1-np.cumprod(1-a,axis=0)),None,1000,0,20)
  self.assertEqual(j.lead,3);self.assertAlmostEqual(j.retention[0,0],2.5);self.assertEqual(j.priority[0,0],0)
  e=Fake(p,3)
  for E in range(2):e.owned[3,E]=e.free.pop()
  t=ThroughputPrefetch(p,e,{3:2},lambda:j.priority,retention_priority=lambda:j.retention,clock=lambda:1.)
  t.pump(force=True);self.assertEqual(e.pre[0][2],1)
  j.reset();self.assertFalse(j.priority.any());self.assertFalse(j.retention.any())
 def test_ready_publication_reselects_and_cancels(self):
  # Import actual publication method without constructing an I/O engine.
  previous=sys.modules.get('executor');stub=types.ModuleType('executor');stub.RankExecutor=object;sys.modules['executor']=stub
  try:
   spec=importlib.util.spec_from_file_location('netgain_preload',ROOT/'spark/flash/preload_executor.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
  finally:
   if previous is None:sys.modules.pop('executor',None)
   else:sys.modules['executor']=previous
  p=Pool([3],3,np.zeros((1,3),bool));p.state[0]=[2,2,1];p.wanted[0,2]=True
  def ex():
   e=mod.PreloadExecutor.__new__(mod.PreloadExecutor);e.preloads={3:dict(phase='ready',expert=2,victim=0,slot=7)};e.publish_attempts=0;e.wait_apply={};e.layers={3:(None,None)};e.slot_of={(3,2):7};e.pre_stats={'cancelled':0};e._rel=lambda s:None
   return e
  score=np.array([[0.,1.4,2.8]]);retention=np.array([[2.5,1.4,0.]])
  e=ex();chosen=[]
  class Stop(Exception):pass
  def row(L,E,*args):chosen.append(E);raise Stop()
  e._row=row
  with self.assertRaises(Stop):e.publish_ready(3,p,score,retention=retention)
  self.assertEqual(chosen,[1])
  e=ex();score[0,2]=.1;e.publish_ready(3,p,score,retention=retention)
  self.assertFalse(e.preloads);self.assertEqual(p.state[0,2],0);self.assertEqual(e.pre_stats['cancelled'],1)
if __name__=='__main__':unittest.main()
