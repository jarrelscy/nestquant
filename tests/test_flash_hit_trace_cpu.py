import sys,pathlib,unittest
import numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'spark/flash'))
from hit_trace import HitTrace,Observer
class Tests(unittest.TestCase):
 def test_partition_and_overlapping_eviction(self):
  state=np.array([2,1,1,1,3,0,0,0,0,0,2]);hot=np.zeros(11,bool);hot[0]=True
  yes=lambda idx:np.isin(np.arange(11),idx)
  result=HitTrace.classify(hot,state,yes([6]),yes([7]),~yes([9]),yes([1]),yes([2]),yes([1,5]),yes([4]))
  self.assertEqual(result.tolist(),['hot','delivery_read_or_queue','delivery_mailbox','delivery_pending_unresolved','demotion_pending_ack','evicted_before_reuse','selected_not_admitted','policy_excluded','forecast_rank_miss','startup_or_uncovered_horizon','cpu_hot_gpu_cold_mismatch'])
 def test_actual_gpu_hot_overrules_cpu_pending(self):
  z=np.zeros(1,bool);o=~z
  self.assertEqual(HitTrace.classify(o,np.array([1]),o,z,o,o,z,z,z)[0],'hot')
 def test_failed_delivery(self):
  z=np.zeros(1,bool);o=~z
  self.assertEqual(HitTrace.classify(z,np.array([0]),o,z,o,z,z,z,o)[0],'delivery_failed')
 def test_observer_fanout(self):
  class O:
   def __init__(self):self.calls=[]
   def event(self,*args):self.calls.append(args)
   def issued_ops(self,*args):self.calls.append(args)
  a,b=O(),O();o=Observer(a,b);o.event('landed',3,8);o.issued_ops([(3,8)],[(3,1)])
  self.assertEqual(a.calls,b.calls);self.assertEqual(len(a.calls),2)
if __name__=='__main__':unittest.main()
