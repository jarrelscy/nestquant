"""Arrival-aware p/F scheduling with published local expert-error benefits.

Expected repeated-use value uses sum p * conditional measured benefit. CDF
supplies a separate configurable first-reuse risk premium; this is a heuristic
risk-averse policy, not a claim of globally optimal regret or LM loss reduction.
"""
import time,math
import numpy as np
from jmt_joint import JointPlanner

class ProbabilityPlanner(JointPlanner):
 def __init__(self,pool,budgets,fixed,benefit,clock=time.monotonic,risk=.5):
  super().__init__(pool,budgets,fixed,clock=clock,mix=1,full_horizon=True,reload_penalty=risk)
  self.benefit=np.asarray(benefit,float)
  if self.benefit.shape!=pool.state.shape or not np.isfinite(self.benefit).all() or np.any(self.benefit<0):raise ValueError('Invalid benefits')
 def reset(self):
  super().reset()
  self.retention=np.zeros_like(self.priority)
 def plan(self,forecasts,ema,rate,reads,tps):
  p=np.asarray(forecasts['activation'],float);cdf=np.asarray(forecasts['first_use'],float)
  expected=(16,*self.pool.state.shape)
  if p.shape!=expected or cdf.shape!=expected:raise ValueError('Invalid probability shape')
  if not np.isfinite(p).all() or not np.isfinite(cdf).all() or np.any((p<0)|(p>1)) or np.any((cdf<0)|(cdf>1)):raise ValueError('Invalid probability range')
  if np.any(np.diff(cdf,axis=0)<-1e-6) or not np.allclose(p[0],cdf[0],atol=1e-6):raise ValueError('Invalid cumulative contract')
  now=self.clock()
  while self.latencies and now-self.latencies[0][0]>8:self.latencies.popleft()
  self.delivery_observed=float(np.percentile([x for _,x in self.latencies],90)) if self.latencies else None
  seconds=max(self.delivery_observed if self.delivery_observed is not None else .04,(reads+1)/rate if rate and rate>0 else .04)
  self.lead=max(1,math.ceil(seconds*max(1.,tps)));self.horizon_end=16
  resident=self.pool.state==2;pending=self.pool.state==1
  value=p*self.benefit[None]
  if self.lead>16:
   # Recovery without claiming a forecast beyond sixteen tokens. No routing EMA.
   incoming=value.mean(0);keep=incoming*1.25
   self.forecast_exhausted+=1;self.fallback_plans+=1;self.horizon_end=0
   risk=np.zeros_like(incoming)
  else:
   incoming=value[self.lead-1:].sum(0);keep=value.sum(0)
   # Probability of first use strictly BEFORE estimated arrival. The first
   # forecast is n+1, so lead1 has no uncovered forecast position.
   risk=(cdf[self.lead-2] if self.lead>1 else 0.)*self.benefit*self.reload_penalty
  score=np.where(resident,keep+risk,incoming)
  required=self.fixed|pending
  score=np.where(self.pool.state==3,-np.inf,score);score=np.where(required,np.inf,score)
  wanted=np.zeros_like(resident)
  for i,b in enumerate(self.budgets):
   if required[i].sum()>b:raise RuntimeError('Fixed/inflight exceed active budget')
   order=np.lexsort((~resident[i],-score[i]));wanted[i,order[:b]]=True
  self.priority=incoming;self.retention=keep+risk;self.plans+=1
  self.cdf_marginal_inconsistency=float((cdf+1e-6<p).mean())
  return wanted
 def stats(self):
  return dict(super().stats(),mode='activation_firstuse_benefit',routing_ema_mix=0.,benefit=('unit_value_per_activation' if np.all(self.benefit==1) else 'positive_mean_local_teacher_error_reduction'),cdf_risk_coefficient=self.reload_penalty,cdf_below_activation_fraction=getattr(self,'cdf_marginal_inconsistency',None))
