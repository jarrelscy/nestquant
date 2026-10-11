"""Flash-only spare-slot preload. Payloads stay unpublished until a paired switch.

All real-table staging is enqueued on the owning compute stream immediately before
its mailbox apply. Old slots are reclaimed only after the existing GPU ack.
"""
import time
import numpy as np
import torch
from executor import RankExecutor

class PreloadExecutor(RankExecutor):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  # The I/O engine may only publish into these private dummy mailboxes.
  sample=next(iter(self.layers.values()))[1]
  self.pre_stage=torch.empty((len(self.layers),*sample.stage.shape),dtype=torch.int64,device=sample.stage.device)
  self.pre_seq=torch.zeros((len(self.layers),*sample.seq.shape),dtype=torch.int32,device=sample.seq.device)
  self.pre_li={L:i for i,L in enumerate(self.layers)}
  self.preloads={};self.pre_tags={};self.pre_stats=dict(issued=0,ready=0,switched=0,cancelled=0)
  self.pre_times=[];self.ready_breakdowns=[];self.publish_attempts=0;self.no_victim_attempts=0
 @property
 def staged_layers(self):return set(self.preloads)
 def preload(self,L,E,victim,sched):
  if L in self.preloads or not self.free:raise RuntimeError('No spare preload capacity')
  i=sched.li[L]
  if sched.state[i,E]!=0 or sched.state[i,victim]!=2:raise RuntimeError('Invalid preload ownership')
  if (L,E) in self.slot_of:raise RuntimeError('Preload already owns a slot')
  slot=self.free.pop();self.slot_of[L,E]=slot;self.tag+=1;tag=self.tag
  sched.state[i,E]=1
  tx=dict(expert=E,victim=victim,slot=slot,tag=tag,phase='reading',issued=time.monotonic())
  self.preloads[L]=tx;self.pre_tags[tag]=L
  if sched.observer is not None:sched.observer.issued_ops([(L,E)],[],preload=True)
  li=self.pre_li[L];stage=self.pre_stage[li,E].data_ptr();seq=self.pre_seq[li,E:E+1].data_ptr()
  self.eng.upgrade(tag,self.rf.rec(L,E),self.slot0+slot*self.rb,stage,self._row(L,E,4,slot),seq,1)
  self.ops[tag]=(L,E,4,1);self.up_tag[L,E]=tag;self.pre_stats['issued']+=1
 def cancel_up(self,L,E,sched=None):
  tx=self.preloads.get(L)
  if tx is not None and tx['expert']==E:return False # paired preload owns slot until completion/ack
  return super().cancel_up(L,E,sched)
 def poll(self,sched=None,issue=True):
  if self.xep or self.xpend:raise RuntimeError('Preload executor does not support borrowed pools')
  for tag,hit,trd,te2e in self.eng.poll():
   L,E,kind,q=self.ops.pop(tag);self.odst.pop(tag,None)
   if kind==4:self.up_tag.pop((L,E),None)
   if tag in self.pre_tags:
    self.pre_tags.pop(tag);tx=self.preloads[L]
    if trd<0:
     self._rel(self.slot_of.pop((L,E)));del self.preloads[L];self.n_failed+=1
     if sched is not None:sched.failed(L,E,read_error=True)
     continue
    tx['phase']='ready';tx['ready']=time.monotonic();self.pre_stats['ready']+=1
    if hasattr(self,'delivery_meter'):tx['ready_snapshot']=self.delivery_meter.snapshot()
    self.n_landed+=1;self.lat.append(te2e);self.lat_w.append(te2e);self.lat_rc.append((trd,te2e-trd))
    continue
   if trd<=-1e8:
    self.n_cancel+=1;self._rel(self.slot_of.pop((L,E)))
    if sched is not None:sched.cancelled(L,E)
    continue
   if trd<0:
    self.n_failed+=1;self._rel(self.slot_of.pop((L,E)))
    if sched is not None:sched.failed(L,E,read_error=True)
    continue
   self.layers[L][1].hseq[E]=q;self.wait_apply[L,E]=(kind,q)
   if kind==4:self.lat.append(te2e);self.lat_w.append(te2e);self.lat_rc.append((trd,te2e-trd));self.n_landed+=1
  for (L,E),(kind,q) in list(self.wait_apply.items()):
   if self.ah[L][E]!=q:continue
   del self.wait_apply[L,E]
   if kind==4:
    if sched is not None:sched.landed(L,E)
   else:
    self._rel(self.slot_of.pop((L,E)))
    if sched is not None:sched.released(L,E)
  for L,tx in list(self.preloads.items()):
   if tx['phase']=='switching' and (L,tx['expert']) not in self.wait_apply and (L,tx['victim']) not in self.wait_apply:
    self.pre_times.append((tx['ready']-tx['issued'],tx['switch']-tx['ready'],time.monotonic()-tx['switch']))
    if len(self.pre_times)>2048:self.pre_times=self.pre_times[-1024:]
    del self.preloads[L] # releases pinned row/seq buffers only after GPU acknowledged
  if issue and self.pend and self.free:
   n=min(len(self.pend),len(self.free));go,self.pend=self.pend[:n],self.pend[n:];self.apply(go,[],sched)
 def publish_ready(self,L,sched,priority,retention=None):
  tx=self.preloads.get(L)
  if tx is None or tx['phase']!='ready':return
  self.publish_attempts+=1
  if 'first_visit' not in tx:tx['first_visit']=time.monotonic()
  i=sched.li[L];E=tx['expert'];victim=tx['victim']
  if not sched.wanted[i,E]:
   # Payload only touched private staging; it was never visible to a kernel.
   self._rel(self.slot_of.pop((L,E)));sched.cancelled(L,E);del self.preloads[L];self.pre_stats['cancelled']+=1;return
  victims=np.flatnonzero((sched.state[i]==2)&~sched.wanted[i])
  if not len(victims):
   self.no_victim_attempts+=1;tx['no_victim']=tx.get('no_victim',0)+1;return
  if retention is not None:
   victim=int(victims[np.argmin(retention[i,victims])])
   if priority[i,E]<=retention[i,victim]:
    self._rel(self.slot_of.pop((L,E)));sched.cancelled(L,E);del self.preloads[L];self.pre_stats['cancelled']+=1;return
  elif victim not in victims:victim=int(victims[np.argmin(priority[i,victims])])
  if (L,victim) in self.wait_apply or (L,E) in self.wait_apply:raise RuntimeError('Conflicting paired publication')
  MB=self.layers[L][1];pairs=[(victim,2,self._row(L,victim,2)),(E,4,self._row(L,E,4,tx['slot']))]
  rows=torch.stack([r for _,_,r in pairs]).pin_memory();seq=torch.tensor([MB.hseq[e]+1 for e,_,_ in pairs],dtype=torch.int32,device='cpu').pin_memory()
  # Called before mailbox.apply on this same compute stream. Both row+seq
  # updates finish before ANY mailbox reader or subsequent expert kernel.
  for k,(e,kind,_) in enumerate(pairs):
   MB.stage[e].copy_(rows[k],non_blocking=True);MB.seq[e:e+1].copy_(seq[k:k+1],non_blocking=True)
   MB.hseq[e]=int(seq[k]);self.wait_apply[L,e]=(kind,MB.hseq[e])
  sched.state[i,victim]=3
  if sched.observer is not None:sched.observer.issued_ops([],[(L,victim)],preload=True)
  tx.update(victim=victim,phase='switching',switch=time.monotonic(),pins=(rows,seq));self.pre_stats['switched']+=1
  if hasattr(self,'delivery_meter') and 'ready_snapshot' in tx:
   self.ready_breakdowns.append(dict(phases=self.delivery_meter.delta(tx['ready_snapshot']),ready_to_first_visit=tx['first_visit']-tx['ready'],first_visit_to_switch=tx['switch']-tx['first_visit'],no_victim_attempts=tx.get('no_victim',0)))
   if len(self.ready_breakdowns)>2048:self.ready_breakdowns=self.ready_breakdowns[-1024:]
 def publish_all_ready(self,sched,priority,retention=None):
  """Caller owns the target compute stream; no side-stream table mutation.

  At a boundary between layers, earlier table consumers precede these copies
  and mailbox kernels on the same stream; future consumers follow them.
  Host reclaim still waits for GPU ack. This also orders queued trace gathers.
  """
  published=0
  for L in [L for L,tx in self.preloads.items() if tx['phase']=='ready']:
   self.publish_ready(L,sched,priority,retention=retention)
   tx=self.preloads.get(L)
   if tx is not None and tx['phase']=='switching':
    self.layers[L][1].apply();published+=1
  return published
 def io_stats(self,name='default'):
  d=super().io_stats(name)
  if hasattr(self,'preloads'):
   d['preload']=dict(self.pre_stats,pending=len(self.preloads),phases={p:sum(t['phase']==p for t in self.preloads.values()) for p in ['reading','ready','switching']})
   if self.pre_times:
    a=np.asarray(self.pre_times)*1000
    d['preload']['phase_p90_ms']=dict(zip(['issue_to_ready','ready_to_switch','switch_to_host_ack'],np.percentile(a,90,axis=0).tolist()))
  if hasattr(self,'delivery_meter'):
   d['delivery_timing']=self.delivery_meter.stats()
   if getattr(self,'target_stream_ms',None):d['delivery_timing']['target_stream_elapsed_ms_including_host_launch_gaps']=dict(mean=float(np.mean(self.target_stream_ms)),p90=float(np.percentile(self.target_stream_ms,90)))
   d['delivery_timing'].update(publish_attempts=self.publish_attempts,no_victim_attempts=self.no_victim_attempts,samples=len(self.ready_breakdowns))
   if self.ready_breakdowns:
    keys=set().union(*(x['phases'] for x in self.ready_breakdowns))
    d['delivery_timing']['ready_wait_phase_mean_ms']={k:float(np.mean([x['phases'].get(k,0.) for x in self.ready_breakdowns]))*1000 for k in keys}
    for key in ['ready_to_first_visit','first_visit_to_switch']:
     vals=np.array([x[key] for x in self.ready_breakdowns])*1000
     d['delivery_timing'][key+'_ms']=dict(mean=float(vals.mean()),p90=float(np.percentile(vals,90)))
  return d
 def busy(self):return bool(self.preloads) or super().busy()
