"""Committed-history jMT using the tested jT incremental trunk implementation.

The latest committed predictor input is n; the next target row is n+1.
Pool forecasts average normalized per-horizon shares, never raw logits.
"""
import math,os
import torch
from torch import nn
from pathlib import Path
from jmt_feedback import FeedbackTuner
from jmt_heads import BatchedHeads
from jt_runtime import IncrementalJT,module

class FuturePool(nn.Module):
 def __init__(self,projection,down,up,bias,first=3,width=3):
  super().__init__();self.projection=projection;self.down=down;self.up=up;self.bias=bias
  self.out_features=projection.out_features;self.first=first;self.width=width
  self.batched=BatchedHeads(projection,down,up,bias)
 def forward(self,x):
  last=min(16,self.first+self.width-1)
  z=self.batched(x,self.first,last)
  shares=z.view(last-self.first+1,*x.shape[:-1],42,288).float().softmax(-1)
  if os.environ.get("NQ_FLASH_HORIZON_EVICT","0")=="1" or os.environ.get("NQ_FLASH_JMT_JOINT","0")=="1":
   if x.shape[0]!=1:raise ValueError('Joint scheduler requires one request')
   logits=self.batched(x[:,-1:],1,16).view(16,42,288)
   self.forecasts=logits.float().softmax(-1).detach().cpu().numpy()
  return shares.mean(0).clamp_min(1e-30).log().flatten(-2)

class JMTIncremental(IncrementalJT):
 def __new__(cls,root,device,dtype):
  if (Path(root)/'jMT_activation.json').exists():
   from jmt_activation import ActivationIncremental
   return ActivationIncremental(root,device,torch.float32)
  return super().__new__(cls)
 def __init__(self,root,device,dtype):
  root=Path(root);api=module(root/'model.py','nq_flash_jmt_model');m=api.load(root,device=device,dtype=dtype)
  self.mode=os.environ.get('NQ_FLASH_JMT_MODE','adaptive')
  if self.mode not in ('adaptive','fixed'):raise ValueError('Unknown jMT scheduler mode')
  first=int(os.environ.get('NQ_FLASH_JMT_FIRST','3'));width=int(os.environ.get('NQ_FLASH_JMT_WIDTH','3'))
  if not 1<=first<=16 or not 1<=width<=16:raise ValueError('Invalid jMT horizon')
  self.delivery_seconds=None;self.initial_first=first;self.base_first=first
  self.feedback=FeedbackTuner() if os.environ.get("NQ_FLASH_JMT_FEEDBACK","1")=="1" else None
  m.bhead=FuturePool(m.bhead,m.down,m.up,m.offset_bias,first,width)
  self.pool=m.bhead
  rope=module(root/'jt_model.py','nq_flash_jmt_rope').rope
  super().__init__(m,rope)
 def reset(self):
  super().reset()
  self.pool.first=self.initial_first;self.base_first=self.initial_first;self.delivery_seconds=None;self.pool.forecasts=None
  if self.feedback is not None:
   self.feedback.reset();self.pool.first,self.pool.width=self.feedback.choice(self.initial_first)
 def delivery(self,records_per_second,pending,tps):
  if self.mode!='adaptive' or not records_per_second or records_per_second<=0:return
  # Queue-service estimate, not a measured end-to-end latency percentile.
  tau=(max(0,pending)+1)/records_per_second
  self.delivery_seconds=tau if self.delivery_seconds is None else .8*self.delivery_seconds+.2*tau
  h=max(1,min(16,math.ceil(1+self.delivery_seconds*max(1,tps))))
  self.base_first=h
  self.pool.first,self.pool.width=self.feedback.choice(h) if self.feedback is not None else (h,self.pool.width)
 def observe(self,count,total,hot,late,seconds):
  if self.feedback is not None:
   self.feedback.observe(count,total,hot,late,seconds)
   self.pool.first,self.pool.width=self.feedback.choice(self.base_first)
 def stats(self):
  return dict(feedback=None if self.feedback is None else self.feedback.stats(),base_horizon=self.base_first,mode=self.mode,first=self.pool.first,last=min(16,self.pool.first+self.pool.width-1),estimated_delivery_seconds=self.delivery_seconds,forecast_origin='last committed input n; next target is n+1')
