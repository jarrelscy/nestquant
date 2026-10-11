"""Observational committed-route trace. Never changes pool selection or I/O.

'Forecast miss' means outside a layer's top-budget rank, not probability zero.
CPU state can lag GPU mailbox activation; record both, never infer hot from CPU.
"""
import os,time,json,gzip,pickle
from pathlib import Path
from collections import Counter
import numpy as np

class Observer:
 def __init__(self,trace,delegate):self.trace=trace;self.delegate=delegate
 def issued_ops(self,ups,downs,*,preload=False):
  self.trace.issued_ops(ups,downs)
  if self.delegate:
   if preload:self.delegate.issued_ops(ups,downs,preload=True)
   else:self.delegate.issued_ops(ups,downs)
 def event(self,kind,L,E):
  self.trace.event(kind,L,E)
  if self.delegate:self.delegate.event(kind,L,E)

class HitTrace:
 def __init__(self,root,pool,executor,budgets):
  self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True);os.chmod(self.root,0o700)
  self.pool=pool;self.ex=executor;self.budgets=budgets;self.request=-1;self.stream=None;self.completed=[]
  self.position=0;self.active=False
 def reset(self,request_id=None):
  if self.active:
   self.stream.close();self.completed.append(self.stats());self.save('requests.json',self.completed)
  self.request_id=request_id;self.request+=1;self.active=True;self.position=0;self.frames={};self.table={}
  sh=self.pool.state.shape;self.issue=np.full(sh,-1,np.int64);self.release=np.full(sh,-1,np.int64);self.failure=np.full(sh,-1,np.int64)
  self.initial_hot=self.pool.state==2;self.ever_hot=self.initial_hot.copy()
  self.counters=Counter();self.events=Counter();self.last_write=0
  p=self.root/f'request-{self.request:03d}.pickle.gz';self.stream=gzip.open(p,'wb',compresslevel=1);os.chmod(p,0o600)
 def save(self,name,x):
  p=self.root/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
 def issued_ops(self,ups,downs):
  if not self.active:return
  for L,E in ups:self.issue[self.pool.li[L],E]=self.position
  self.events['issued']+=len(ups);self.events['demotions']+=len(downs)
 def event(self,kind,L,E):
  if not self.active:return
  i=self.pool.li[L];self.events[kind]+=1
  if kind=='landed':self.ever_hot[i,E]=True
  elif kind=='released':self.release[i,E]=self.position
  elif kind=='failed':self.failure[i,E]=self.position
 def begin(self,predictor,prefill):
  self.frames={};self.table={};self.prefill=prefill;self.origin=predictor.t-1
  pool=getattr(predictor.net,'pool',None);forecasts=getattr(pool,'forecasts',None)
  if isinstance(forecasts,dict):self.forecast=forecasts['activation'];self.kind='activation_probability';self.max_h=16
  elif forecasts is not None:self.forecast=np.asarray(forecasts);self.kind='salience_share';self.max_h=16
  elif predictor.block is not None:self.forecast=np.broadcast_to(predictor.block,(15,*predictor.block.shape));self.kind='jt_block_share';self.max_h=15
  else:self.forecast=None;self.kind='none';self.max_h=0
 def layer(self,L):
  i=self.pool.li[L];state=self.pool.state[i].copy();ne=len(state)
  mailbox=np.array([(L,e) in self.ex.wait_apply for e in range(ne)],bool)
  reading=np.zeros(ne,bool)
  for l,e,lv,_ in self.ex.ops.values():
   if l==L and lv==4:reading[e]=True
  slots=np.array([self.ex.slot_of.get((L,e),-1) for e in range(ne)],np.int64)
  self.frames[L]=dict(state=state,desired=self.pool.wanted[i].copy(),mailbox=mailbox,reading=reading,slot=slots,
    issue=self.issue[i].copy(),release=self.release[i].copy(),failure=self.failure[i].copy(),ever_hot=self.ever_hot[i].copy())
 @staticmethod
 def classify(hot,state,desired,rank_hit,covered,reading,mailbox,evicted,failed):
  # Earlier assignments take precedence; overlapping history flags are saved.
  out=np.full(hot.shape,'forecast_rank_miss',dtype='U32')
  out[rank_hit]='policy_excluded';out[~covered]='startup_or_uncovered_horizon'
  out[desired]='selected_not_admitted';out[failed]='delivery_failed'
  out[evicted]='evicted_before_reuse';out[state==3]='demotion_pending_ack'
  out[state==1]='delivery_pending_unresolved';out[(state==1)&reading]='delivery_read_or_queue'
  out[(state==1)&mailbox]='delivery_mailbox'
  out[state==2]='cpu_hot_gpu_cold_mismatch';out[hot]='hot'
  return out
 def consume(self,rows,hot,gpu_tables):
  if not rows:return
  n=len(rows);positions=np.array([x[0] for x in rows]);ids=np.stack([x[2] for x in rows]);sh=ids.shape
  rank=np.full(sh,-1,np.int16);score=np.full(sh,np.nan,np.float32);covered=np.zeros(sh,bool)
  h=positions-self.origin
  if self.forecast is not None:
   for k,offset in enumerate(h):
    if 1<=offset<=self.max_h:
     a=self.forecast[offset-1];order=np.argsort(-a,axis=-1,kind='stable');r=np.empty_like(order)
     np.put_along_axis(r,order,np.broadcast_to(np.arange(a.shape[-1]),a.shape),axis=-1)
     rank[k]=np.take_along_axis(r,ids[k],axis=-1);score[k]=np.take_along_axis(a,ids[k],axis=-1);covered[k]=True
  def gather(key):return np.stack([self.frames[L][key][ids[:,i]] for i,L in enumerate(self.pool.layers)],axis=1)
  state=gather('state');desired=gather('desired');release=gather('release');issue=gather('issue');failure=gather('failure')
  evicted=(release>=0)&((positions[:,None,None]-release)<16)&~hot
  failed=(failure>=0)&(failure>=issue)&~hot
  rank_hit=(rank>=0)&(rank<np.asarray(self.budgets)[None,:,None])
  categories=self.classify(hot,state,desired,rank_hit,covered,gather('reading'),gather('mailbox'),evicted,failed)
  # Inspect level and actual GU/down record addresses on the GPU's execution
  # stream. CPU state alone is insufficient while acknowledgments are pending.
  tab=gpu_tables[:n];slots=gather('slot');base=self.ex.slot0+slots*self.ex.rb
  gu=base+self.ex.lay['seg']['gu.p4'][0];dn=base+self.ex.lay['seg']['dn.p4'][0]
  bad_slot=hot&((slots<0)|(tab[...,1]!=gu)|(tab[...,2]!=dn))
  phase='prefill' if self.prefill else 'decode'
  for k,v in Counter(categories.reshape(-1)).items():self.counters[phase+':'+k]+=v
  self.counters[phase+':slot_pointer_mismatch']+=int(bad_slot.sum())
  self.counters[phase+':evicted_overlap']+=int(evicted.sum())
  self.counters[phase+':tokens']+=n
  pickle.dump(dict(request=self.request,phase=phase,positions=positions,expert_ids=ids,forecast_kind=self.kind,forecast_origin=self.origin,
    forecast_score=score,forecast_rank=rank,hot=hot,state=state,desired=desired,category=categories,evicted_within16=evicted,
    issue_position=issue,release_position=release,slot_pointer_mismatch=bad_slot),self.stream,protocol=5)
  self.position=int(positions[-1])+1
  if time.monotonic()-self.last_write>2:
   self.save('current.json',self.stats());self.stream.flush();self.last_write=time.monotonic()
 def stats(self):return dict(request=self.request,request_id=self.request_id,position=self.position,counts=dict(self.counters),events=dict(self.events),definition='Mutually exclusive immediate cause plus overlapping16token eviction flag. Rank cutoff is active pool budget; positive forecast is not a binary prediction. Pointer mismatch is a diagnostic flag, not proof of bad record contents.')
