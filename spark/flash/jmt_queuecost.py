"""Joint admission/eviction in a common expected-hit currency.

Adaptation of TIP's cost/benefit principle, not an implementation of TIP-2 or
an optimal stochastic cache. Finite-horizon, one-exchange approximation:
we charge ALL forecast uses lost after eviction, rather than assume a free
future reload. This conservatively includes reuse loss within the horizon;
reload traffic is charged when a subsequent replacement is actually admitted.

Paired preloads keep the resident until arrival. Immediate demotions do not.
Ready payloads no longer pay an arrival delay. These three decisions must not
share one scalar score. Queue delay/throughput estimates and physical limits
are supplied by the existing bounded prefetcher.
"""
import numpy as np
from jmt_probability_planner import ProbabilityPlanner

class QueueCostPlanner(ProbabilityPlanner):
 def reset(self):
  super().reset()
  self.paired_retention=np.zeros_like(self.priority)
  self.publication_priority=np.zeros_like(self.priority)
  self.publication_retention=np.zeros_like(self.priority)
 def plan(self,forecasts,ema,rate,reads,tps):
  # Validate contract, update measured end-to-end delay, and retain the
  # immediate-demotion loss/risk calculation and exhausted-horizon fallback.
  old=super().plan(forecasts,ema,rate,reads,tps)
  value=np.asarray(forecasts['activation'],float)*self.benefit[None]
  self.publication_priority=value.sum(0)
  self.publication_retention=value.sum(0)
  if self.lead>16:
   self.paired_retention=self.retention.copy()
   return old
  self.paired_retention=self.priority.copy()
  # Choose the prospective cache AFTER the queued load arrives. Early uses of
  # a paired victim are still served, so charging them here is double counting.
  # Immediate demotions retain the stricter full-horizon cost in self.retention.
  required=self.fixed|(self.pool.state==1)
  score=np.where(self.pool.state==3,-np.inf,self.priority)
  score=np.where(required,np.inf,score)
  wanted=np.zeros_like(self.pool.wanted)
  resident=self.pool.state==2
  for i,b in enumerate(self.budgets):
   order=np.lexsort((~resident[i],-score[i]));wanted[i,order[:b]]=True
  return wanted
 def stats(self):
  return dict(super().stats(),mode='queue_cost_benefit',eviction_cost='all_expected_uses_lost_after_swap',paired_prearrival_loss_charged=False,ready_payload_delay_charged=False,optimality_claim=False)
