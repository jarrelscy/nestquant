import pathlib,sys,unittest
import numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'spark/flash'))
from nq_flash_pool import Pool
from nq_flash_prefetch import ThroughputPrefetch

class Executor:
 def __init__(self,pool,slots):
  self.p=pool;self.free=list(range(slots));self.ops={};self.owned={};self.reading=set();self.rb=1000;self.rate=.0001
 def io_stats(self,name):return {'delivered_GBps':self.rate}
 def apply(self,ups,downs,sched):
  for k in downs:
   assert k not in self.ops
   self.ops[k]=3
  for k in ups:
   assert k not in self.ops and self.free
   self.owned[k]=self.free.pop();self.ops[k]=1
 def cancel_up(self,L,E,sched):
  k=(L,E)
  if k in self.reading:return True
  if k not in self.ops:return False
  del self.ops[k];self.free.append(self.owned.pop(k));sched.cancelled(L,E);return True
 def complete(self,k):
  kind=self.ops.pop(k);self.reading.discard(k)
  if kind==1:self.p.landed(*k)
  else:self.free.append(self.owned.pop(k));self.p.released(*k)

class Tests(unittest.TestCase):
 def make(self):
  wanted=np.zeros((2,8),bool);wanted[:,:4]=1
  p=Pool([3,44],8,wanted);e=Executor(p,10);clock=[0.]
  t=ThroughputPrefetch(p,e,{3:4,44:4},lambda:np.tile(np.arange(8),(2,1)),max_pending=4,min_pending=2,clock=lambda:clock[0])
  return p,e,t,clock
 def test_bounded_admission_converges_without_exceeding_layer_budget(self):
  p,e,t,clock=self.make()
  for phase in range(10):
   p.wanted[:]=False;p.wanted[:,phase%5:phase%5+4]=True
   for step in range(30):
    clock[0]+=.02;t.pump(force=True)
    self.assertLessEqual(len(e.ops),4)
    self.assertTrue(np.all(np.count_nonzero(p.state,axis=1)<=4))
    for k in list(e.ops):e.complete(k)
    if np.all((p.state==2)==p.wanted):break
   else:self.fail('did not converge')
   self.assertEqual(len(e.free),2)
 def test_cancellation_never_reuses_inflight_slot(self):
  p,e,t,clock=self.make();t.pump(force=True)
  k=next(iter(e.ops));e.reading.add(k);slot=e.owned[k]
  p.wanted[:]=False;t.pump(force=True)
  self.assertEqual(e.owned[k],slot);self.assertNotIn(slot,e.free)
  self.assertEqual(p.state[p.li[k[0]],k[1]],1)
  e.complete(k)
  self.assertEqual(p.state[p.li[k[0]],k[1]],2)
  self.assertNotIn(slot,e.free)
 def test_bandwidth_changes_budget_but_never_exceeds_cap(self):
  p,e,t,clock=self.make();e.rate=.000001;t.measure(0)
  self.assertEqual(t.limit,2)
  e.rate=1;t.peak_pending=4;t.measure(1)
  self.assertEqual(t.limit,4)
 def test_cancels_only_stale_and_prioritizes_wanted(self):
  p,e,t,clock=self.make();t.pump(force=True)
  self.assertEqual(set(e.ops),{(3,3),(44,3),(3,2),(44,2)})
  p.wanted[:]=False;p.wanted[:,0]=True;t.pump(force=True)
  self.assertEqual(t.cancellations,4)
  self.assertTrue(all(k[1]==0 for k in e.ops))
 def test_residents_not_all_evicted_on_pool_change(self):
  p,e,t,clock=self.make()
  for _ in range(10):
   t.pump(force=True)
   for k in list(e.ops):e.complete(k)
  p.wanted[:,:4]=False;p.wanted[:,4:]=True
  t.pump(force=True)
  self.assertLessEqual(np.count_nonzero(p.state==3),2)
  self.assertGreaterEqual(np.count_nonzero(p.state==2),6)
if __name__=='__main__':unittest.main()
