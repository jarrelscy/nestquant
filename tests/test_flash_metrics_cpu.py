import json
import pathlib
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'spark/flash'))
from nq_flash_metrics import RoutingMetrics

class MetricsTests(unittest.TestCase):
 def test_rejected_rows_separate_and_zero_weight_is_not_active(self):
  with tempfile.TemporaryDirectory() as d:
   m=RoutingMetrics([3,44],pathlib.Path(d)/'stats.json');m.new_request()
   hot=np.zeros((3,2,8),bool);active=np.ones_like(hot)
   hot[0,:,:2]=1;hot[1,:,:4]=1;hot[2,:,:]=1
   active[0,0,0]=0
   m.add(hot,active,2,'decode');m.export(True)
   r=json.loads(m.path.read_text())['total']
   self.assertEqual(r['decode_committed']['hot_expert_activations'],11)
   self.assertEqual(r['decode_committed']['expert_activations'],31)
   self.assertEqual(r['decode_committed']['mean_hot_of_8'],2.75)
   self.assertEqual(r['decode_executed']['hot_expert_activations'],27)
   self.assertEqual(r['decode_executed']['hot_count_histogram'][8],2)
   m.new_request();self.assertEqual(m.current,{})
   self.assertEqual(m.summarize(m.total)['decode_committed']['hot_expert_activations'],11)
 def test_desired_misses_and_late_loads_partition_actual_misses(self):
  m=RoutingMetrics([3],'/unused');m.new_request()
  hot=np.array([[[1,1,0,0,1,1,0,0]]],bool)
  desired=np.array([[[1,0,1,0,1,0,1,0]]],bool)
  active=np.ones_like(hot)
  m.add(hot,active,1,'decode',desired=desired)
  r=m.summarize(m.total)['decode_committed']
  self.assertEqual(r['hit_rate'],.5)
  self.assertEqual(r['desired_pool_hit_rate'],.5)
  self.assertEqual(r['desired_but_cold'],2)
  self.assertEqual(r['outside_desired_and_cold'],2)
  self.assertEqual(r['desired_but_cold_rate']+r['outside_desired_and_cold_rate'],1-r['hit_rate'])
  m.io={'ssd_GBps':1.2}
 def test_prefill_empty_commit_and_all_cold(self):
  m=RoutingMetrics([3],'/unused');m.new_request()
  m.add(np.zeros((2,1,8)),np.ones((2,1,8)),0,'prefill')
  r=m.summarize(m.total)
  self.assertIsNone(r['prefill_committed']['hit_rate'])
  self.assertEqual(r['prefill_executed']['hit_rate'],0)
  self.assertEqual(r['prefill_executed']['hot_count_histogram'][0],2)
  with self.assertRaises(ValueError):m.add(np.zeros((2,1,8)),np.ones((2,1,8)),3,'decode')
 def test_salience_weighting_rejections_and_partition(self):
  m=RoutingMetrics([3,44],'/unused');m.new_request()
  hot=np.zeros((2,2,8),bool);hot[:,:,0]=True
  desired=np.zeros_like(hot);desired[:,:,1]=True
  active=np.ones_like(hot);active[0,1,0]=False
  weights=np.zeros((2,2,8));weights[:,:,0]=2;weights[:,:,1]=1
  xn=np.array([[1,10],[100,1000]])
  salience=weights**2*xn[:,:,None]
  m.add(hot,active,1,'decode',desired=desired,salience=salience)
  r=m.summarize(m.total)
  c=r['decode_committed'];e=r['decode_executed']
  # First layer contributes 4 hot + 1 cold; second contributes only 10 cold.
  self.assertEqual(c['salience_sum'],15)
  self.assertEqual(c['hot_salience_sum'],4)
  self.assertAlmostEqual(c['hot_salience_coverage'],4/15)
  self.assertAlmostEqual(c['desired_salience_coverage'],11/15)
  self.assertAlmostEqual(c['hot_salience_coverage']+c['desired_but_cold_salience_fraction']+c['outside_desired_and_cold_salience_fraction'],1)
  self.assertEqual(e['salience_sum'],5515)
  self.assertEqual(e['hot_salience_sum'],4404)
  self.assertEqual(c['layers']['3']['hot_salience_coverage'],.8)
  self.assertEqual(c['layers']['44']['hot_salience_coverage'],0)
  m.new_request();m.add(hot,active,0,'prefill',salience=np.zeros_like(salience))
  self.assertIsNone(m.summarize(m.current)['prefill_executed']['hot_salience_coverage'])
  self.assertEqual(m.summarize(m.total)['decode_committed']['salience_sum'],15)
 def test_invalid_salience_rejected(self):
  m=RoutingMetrics([3],'/unused');h=np.ones((1,1,8),bool)
  for s in (np.zeros((1,8)),np.full(h.shape,-1.),np.full(h.shape,np.nan)):
   with self.assertRaises(ValueError):m.add(h,h,1,'decode',salience=s)
if __name__=='__main__':unittest.main()
