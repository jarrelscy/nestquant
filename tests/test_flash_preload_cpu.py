import sys,pathlib,unittest,types
import numpy as np
sys.path[:0]=[str(pathlib.Path(__file__).resolve().parents[1]/'spark/flash')]
from nq_flash_pool import Pool
from nq_flash_prefetch import ThroughputPrefetch
from test_flash_prefetch_cpu import Executor
class Fake(Executor):
 def __init__(self,p,slots):super().__init__(p,slots);self.staged_layers=set();self.pre=[]
 def preload(self,L,E,victim,p):
  assert self.free and L not in self.staged_layers
  assert p.state[p.li[L],victim]==2
  self.owned[L,E]=self.free.pop();p.state[p.li[L],E]=1;self.ops[L,E]=(L,E,4,1);self.staged_layers.add(L);self.pre.append((L,E,victim))
class Tests(unittest.TestCase):
 def make(self,spares=2):
  p=Pool([3,4],8,np.array([[0,0,0,0,1,1,1,1]]*2,bool));p.state[:,:4]=2
  ex=Fake(p,8+spares)
  for L in p.layers:
   for E in range(4):ex.owned[L,E]=ex.free.pop()
  pre=ThroughputPrefetch(p,ex,{3:4,4:4},lambda:np.tile(np.arange(8),(2,1)),max_pending=8,min_pending=8,clock=lambda:0.)
  return p,ex,pre
 def test_preload_preserves_victims_and_active_budget(self):
  p,e,t=self.make();t.pump(force=True)
  self.assertEqual(len(e.pre),2);self.assertEqual((p.state==2).sum(),8);self.assertEqual((p.state==3).sum(),0)
  self.assertTrue(np.all((p.state==2).sum(1)==4));self.assertEqual(len(e.free),0)
 def test_no_duplicate_or_early_reuse(self):
  p,e,t=self.make();t.pump(force=True);old=e.owned.copy()
  for _ in range(10):t.pump(force=True)
  self.assertEqual(e.owned,old);self.assertEqual(len(e.pre),2);self.assertEqual(len(set(e.owned.values())),10)
 def test_no_spare_uses_original_safe_path(self):
  p,e,t=self.make(0);t.pump(force=True);self.assertFalse(e.pre);self.assertEqual((p.state==3).sum(),2)
 def test_bootstrap_does_not_stage(self):
  p,e,t=self.make();t.pump(force=True,bootstrap=True);self.assertFalse(e.pre)
 def test_staged_layer_never_demotes_victim_early(self):
  p,e,t=self.make(1);t.pump(force=True)
  L,E,V=e.pre[0];self.assertEqual(p.state[p.li[L],V],2);self.assertFalse(any(op[2]==2 and op[0]==L for op in e.ops.values()))
if __name__=='__main__':unittest.main()
