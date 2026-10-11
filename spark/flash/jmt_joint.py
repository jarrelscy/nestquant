"""Joint causal admission/retention using measured mailbox-ready delay.

Only committed routing updates regret. Pool callbacks report acknowledged table
changes, not SSD completion. No executor ownership or synchronization is changed.
"""
import math,time
from collections import deque
import numpy as np

class JointPlanner:
 def __init__(self,pool,budgets,fixed,clock=time.monotonic,width=8,mix=.75,full_horizon=False,reload_penalty=.5):
  self.pool=pool;self.budgets=np.asarray(budgets,int);self.fixed=np.asarray(fixed,bool)
  if self.fixed.shape!=pool.state.shape or np.any(self.fixed.sum(1)>self.budgets):raise ValueError('Invalid joint fixed pool')
  if not 1<=width<=16 or not 0<=mix<=1:raise ValueError('Invalid joint parameters')
  if not math.isfinite(reload_penalty) or reload_penalty<0:raise ValueError('Invalid reload penalty')
  self.full_horizon=full_horizon;self.reload_penalty=reload_penalty
  self.decode=True;self.clock=clock;self.width=width;self.mix=mix;self.latencies=deque(maxlen=512)
  self.reset()
 def reset(self):
  self.latencies.clear()
  self.position=0;self.loads={};self.replacements={};self.evicted={};self.down_pending=set()
  self.lead=1;self.horizon_end=1;self.plans=0;self.issued=0;self.demotions=0
  self.landed_count=0;self.deadline_late=0;self.regret_events=0;self.regret_mass=0.;self.late_mass=0.;self.total_mass=0.
  self.regret_ema=0.;self.regret_trials=0;self.forecast_exhausted=0;self.fallback_plans=0;self.delivery_observed=None
  self.priority=np.zeros_like(self.pool.state,dtype=float)
 def issued_ops(self,ups,downs,*,preload=False):
  now=self.clock()
  for L,E in downs:
   if not preload:self.replacements[L]=(now,self.position)
   self.demotions+=1;self.down_pending.add((L,E))
  for L,E in ups:
   start,pos=(now,self.position) if preload else self.replacements.pop(L,(now,self.position))
   self.loads[L,E]=(start,pos+self.lead-1,self.position,self.decode);self.issued+=1
 def event(self,kind,L,E):
  key=L,E
  if kind=='landed':
   item=self.loads.pop(key,None)
   if item is not None:
    start,deadline,_,decode=item
    if decode:self.latencies.append((self.clock(),max(0.,self.clock()-start)))
    self.landed_count+=1
    self.deadline_late+=int(self.position>deadline)
  elif kind=='released' and key in self.down_pending:
   self.down_pending.remove(key);self.evicted[key]=self.position
  elif kind=='failed':self.loads.pop(key,None)
 def committed(self,rows,hot=None,decode=True):
  for j,row in enumerate(rows):
   pos,_,ids,w,xn=row
   if pos!=self.position:raise ValueError('Noncausal joint planner input')
   expired=[k for k,t in self.evicted.items() if pos-t>=16]
   for key in expired:
    self.evicted.pop(key);self.regret_ema*=.98;self.regret_trials+=1
   if decode:
    sal=w.astype(float)**2*xn[:,None];self.total_mass+=float(sal.sum())
    for i,L in enumerate(self.pool.layers):
     for a,E in enumerate(ids[i]):
      key=L,int(E);cold=not bool(hot[j,i,a]) if hot is not None else self.pool.state[i,E]!=2
      if key in self.loads and cold:self.late_mass+=float(sal[i,a])
      if key in self.evicted and cold:
       self.regret_events+=1;self.regret_mass+=float(sal[i,a]);self.evicted.pop(key)
       self.regret_ema=.98*self.regret_ema+.02;self.regret_trials+=1
   self.position+=1
 def plan(self,forecasts,ema,rate,reads,tps):
  f=np.asarray(forecasts,float);ema=np.asarray(ema,float)
  if f.shape!=(16,*self.pool.state.shape) or not np.isfinite(f).all() or np.any(f<0) or not np.allclose(f.sum(-1),1,atol=.003):raise ValueError('Invalid joint forecasts')
  if ema.shape!=self.pool.state.shape or not np.isfinite(ema).all() or np.any(ema<0):raise ValueError('Invalid joint EMA')
  # Actual admission -> mailbox acknowledgment includes demotion, queue, SSD,
  # transfer, and layer activation. Queue estimate guards sudden backlog growth.
  now=self.clock()
  while self.latencies and now-self.latencies[0][0]>8.:self.latencies.popleft()
  self.delivery_observed=float(np.percentile([v for _,v in self.latencies],90)) if self.latencies else None
  observed=self.delivery_observed if self.delivery_observed is not None else .04
  queue=(reads+1)/rate if rate and rate>0 else .04
  seconds=max(observed,queue)
  self.lead=max(1,math.ceil(seconds*max(1.,tps)))
  state=self.pool.state;resident=state==2;pending=state==1
  shares=ema/np.maximum(ema.sum(1,keepdims=True),1e-30)
  mixed=self.mix*f+(1-self.mix)*shares[None]
  if self.lead>16:
   # The forecasts do not cover delivery. Continue bounded delivery using the
   # causal routing EMA, rather than freezing admissions (which would prevent
   # new latency observations). Do not relabel horizon16 as a later forecast.
   self.forecast_exhausted+=1;self.fallback_plans+=1;self.horizon_end=0
   incoming=shares
   score=np.where(resident,shares*(1.25+.5*self.regret_ema),shares)
  else:
   end=16 if self.full_horizon else min(16,self.lead+self.width-1);self.horizon_end=end
   incoming=mixed[self.lead-1:end].sum(0)
   keep=mixed[:end].sum(0)
   if self.full_horizon:
    # Full remaining forecast value plus a heuristic reload-risk premium for
    # a resident forecast to matter late. This is a risk penalty, not an extra
    # prediction of benefit. Incoming scores pay a per-record opportunity cost.
    tail_weights=(np.arange(1,17,dtype=float)/16.)**2
    reload_risk=(mixed*tail_weights[:,None,None]).sum(0)*(self.lead/16.)*self.reload_penalty
    record_cost=self.reload_penalty*self.lead/state.shape[1]
    incoming=incoming-record_cost
    score=np.where(resident,keep*(1+.5*self.regret_ema)+reload_risk,incoming)
   else:
    score=np.where(resident,keep*(1+.5*self.regret_ema),incoming)
  required=self.fixed|pending
  score=np.where(state==3,-np.inf,score)
  score=np.where(required,np.inf,score)
  wanted=np.zeros_like(state,bool)
  for i,b in enumerate(self.budgets):
   if required[i].sum()>b:raise RuntimeError('In-flight/fixed pool exceeds budget')
   order=np.lexsort((~resident[i],-score[i]))
   wanted[i,order[:b]]=True
  self.priority=incoming;self.plans+=1
  return wanted
 def stats(self):
  return dict(mode='joint_measured_delivery',lead_tokens=self.lead,horizon_end=self.horizon_end,width=self.width,full_horizon=self.full_horizon,reload_penalty=self.reload_penalty,
   delivery_p90_ms=self.delivery_observed*1000 if self.delivery_observed is not None else None,
   fallback_plans=self.fallback_plans,latency_max_age_seconds=8.,
   delivery_samples=len(self.latencies),issued=self.issued,demotions=self.demotions,landed=self.landed_count,
   late_vs_estimated_deadline=self.deadline_late,regret_first_cold_reuse_16=self.regret_events,
   regret_first_reuse_salience_fraction=self.regret_mass/max(self.total_mass,1e-30),
   pending_load_cold_salience_fraction=self.late_mass/max(self.total_mass,1e-30),
   regret_ema=self.regret_ema,forecast_exhausted=self.forecast_exhausted,plans=self.plans)
