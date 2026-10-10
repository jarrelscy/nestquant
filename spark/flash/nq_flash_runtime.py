"""TP1 Flash executor, release jT policy and committed-row handoff.

No Python streaming thread: failures propagate to serving instead of leaving a
waiter hung. io_uring/H2D remain asynchronous; mailbox acknowledgments own reuse.
The predictor runs on a dedicated CUDA stream. Graphs and prefix restores are
explicitly disabled until routing-history buffers support them.
"""
import os,sys,time,json
from pathlib import Path
import numpy as np
import torch
from nq_flash_layout import validate_repack,tile_config
from nq_flash_pool import Pool
from jt_runtime import RowLedger,CommittedPredictor
from nq_flash_metrics import RoutingMetrics
from nq_flash_prefetch import ThroughputPrefetch
ROOT=Path(os.environ.get('NQ_HOME',Path(__file__).resolve().parents[2]))
sys.path[:0]=[str(ROOT/'streaming'),str(ROOT/'sm120')]

class Runtime:
 def __init__(s):
  s.layers={};s.layer_ids=list(range(3,45));s.started=False;s.dummy=True;s.batch=None;s.rows={}
  s.ledger=RowLedger();s.pace_deadline=0.;s.tps=float(os.environ.get('NQ_FLASH_TPS','0'))
 def add_layer(s,L,device):
  from resident import load
  from moe import MoELayer,Mailbox
  ex,H,I=load(f"{os.environ['NQ_REPACK']}/res/rank0/L{L}.pt",device)
  if (H,I,len(ex))!=(4096,2048,288):raise ValueError(f'Wrong resident dimensions L{L}')
  m=MoELayer(288,H,I,Bmax=8,dev=device)
  if m.M.swiglu_limit()!=10:raise ValueError('Flash kernel must use SwiGLU clamp 10')
  for E,x in ex.items():
   if (x.gu.bk,x.dn.bk,x.gu.rk,x.dn.rk)!=(5,5,2,10):raise ValueError(f'Wrong Flash codes L{L} E{E}')
   m.set(E,x,2)
  m.cfg_gu=tile_config(4096,4096);m.cfg_dn=tile_config(4096,2048)
  s.layers[L]=(m,Mailbox(m),ex)
  if set(s.layers)==set(s.layer_ids):s.start(device)
 def start(s,device):
  from stream_engine import RankFile,unified_default
  from executor import RankExecutor
  validate_repack(os.environ['NQ_REPACK'])
  s.metrics=RoutingMetrics(s.layer_ids,os.environ.get('NQ_FLASH_ROUTING_STATS','/artifacts/nq-flash-routing.json'))
  s.metrics.export(force=True)
  s.predictor=CommittedPredictor(os.environ['NQ_FLASH_MODEL'],os.environ.get('NQ_FLASH_PRESET','spark_128K'),device,n_fixed=int(os.environ.get('NQ_FLASH_FIXED_PER_LAYER','0')),fixed_fraction=os.environ.get('NQ_FLASH_FIXED_FRACTION'))
  s.pool=Pool(s.layer_ids,288,s.predictor.prepare())
  nactive=sum(s.predictor.budgets.values());spares=int(os.environ.get('NQ_FLASH_SPARE_SLOTS','8'))
  s.prefetch=None
  mode=os.environ.get('NQ_FLASH_PREFETCH','throughput')
  if mode not in ('throughput','legacy'):raise ValueError('Unknown Flash prefetch mode')
  s.executor=RankExecutor(RankFile(os.environ['NQ_REPACK'],0),s.layers,nactive+spares,
                         n_host=8,qd=8,device=device.index,unified=unified_default(device),wait_for_slot=(mode=="legacy"))
  if mode=='throughput':
   s.prefetch=ThroughputPrefetch(s.pool,s.executor,s.predictor.budgets,s.predictor.loading_priority,
       max_pending=int(os.environ.get('NQ_FLASH_PREFETCH_MAX_PENDING','64')),
       max_demotions=int(os.environ.get('NQ_FLASH_PREFETCH_MAX_DEMOTIONS','32')),
       lookahead_seconds=float(os.environ.get('NQ_FLASH_PREFETCH_SECONDS','0.1')))
  else:s.executor.apply(*s.pool.operations(),s.pool)
  s.started=True;s.drain();s.executor.io_stats("routing_metrics")
  print(json.dumps({'nq_flash':'ready','active_slots':nactive,'spare_slots':spares,'fixed':s.predictor.n_fixed,'fixed_per_layer':s.predictor.fixed_counts,'fixed_ranking':'fixed_set.json:score','fixed_ids':{str(L):ids.tolist() for L,ids in zip(s.predictor.layers,s.predictor.fixed_ids)},'floating_slots':nactive-s.predictor.n_fixed,
        'record_bytes':s.executor.rb,'host_mapped_slots':s.executor.unified,
        'slot_bytes':(nactive+spares)*s.executor.rb,'predictor_bytes':sum(p.numel()*p.element_size() for p in s.predictor.net.model.parameters())}),flush=True)
 def poll(s):
  s.executor.poll(s.pool)
  if s.prefetch is not None:s.prefetch.pump()
 def dispatch(s):
  s.poll()
  if s.prefetch is None:s.executor.apply(*s.pool.operations(),s.pool)
  else:s.prefetch.pump(force=True)
 def drain(s):
  start=time.monotonic()
  while s.executor.busy() or np.any(s.pool.wanted & (s.pool.state!=2)):
   s.poll()
   if s.prefetch is not None:s.prefetch.pump(force=True,bootstrap=True)
   for _,mb,_ in s.layers.values():mb.apply()
   torch.cuda.synchronize()
   if time.monotonic()-start>300:raise TimeoutError('Flash initial residual load')
   time.sleep(.001)
 def begin(s,input_ids,positions):
  s.rows={}
  if s.dummy:return
  if torch.cuda.is_current_stream_capturing():raise RuntimeError('Flash predictor history requires eager execution')
  if s.batch is None:raise RuntimeError('Flash worker input hook missing')
  s.metric_skip=any(r.startswith(("_warmup_","_v2_mixed_warmup","_dummy_req_")) for r in s.batch.req_ids)
  n=int(s.batch.num_tokens)
  # Worker IDs remain available even when multimodal forward uses inputs_embeds.
  s.tokens=s.batch.input_ids[:n].detach().cpu().numpy().copy()
  s.positions=positions[:n].detach().cpu().numpy().copy()
  if s.positions.ndim!=1:raise ValueError('Unsupported multi-axis positions')
  if len(s.positions) and s.positions[0]==0:
   s.predictor.reset();s.ledger.reset();s.pace_deadline=0.
   if not s.metric_skip:s.metrics.new_request()
  if s.ledger.pending:
   raise RuntimeError('Previous target rows were not resolved by the sampler hook')
  if len(s.positions) and int(s.positions[0])!=s.ledger.committed_end:
   raise RuntimeError('Predictor history discontinuity; prefix restores are disabled')
  if s.tps>0 and not s.batch.has_prefill:
   delay=s.pace_deadline-time.monotonic()
   if delay>0:time.sleep(delay)
  s.metric_phase="prefill" if s.batch.has_prefill else "decode"
  s.step_started=time.monotonic()
  s.pool.wanted=s.predictor.prepare();s.dispatch()
  # Diagnostic only: ensure every selected upgrade is ready before verify.
  # Preserve jT selection and active budgets; pay the delivery delay explicitly.
  if not s.batch.has_prefill and os.environ.get('NQ_FLASH_WAIT_FOR_UPGRADES')=='1':
   s.drain()
 def forward(s,L,x,weights,ids):
  m,mb,_=s.layers[L];s.poll();mb.apply()
  xh=x.half().contiguous();rw=weights.half().contiguous();sel=ids.long().contiguous()
  out=m.prefill(xh,sel,rw) if len(x)>8 else m(xh,sel,rw)
  if not s.dummy:
   n=len(s.tokens)
   # Own the snapshot: routing buffers may be reused by the next layer.
   # IDs are <=287 and exact in float32; one packed D2H at finish avoids
   # three synchronous copies per layer on the eager serving thread.
   s.rows[L]=torch.cat((ids[:n].detach().float(),weights[:n].detach().float(),
                       x[:n].float().square().sum(-1,keepdim=True),
                       (m.table[sel[:n],0]==4).float(),(rw[:n]!=0).float()),dim=-1)
  return out.to(x.dtype)
 def finish(s):
  if s.dummy:return
  if set(s.rows)!=set(s.layer_ids):raise RuntimeError('Incomplete target routing rows')
  packed=torch.stack([s.rows[L] for L in s.layer_ids],1).cpu().numpy()
  s.routing_desired=np.take_along_axis(s.pool.wanted[None,:,:],packed[:,:,:8].astype(np.int64),axis=2)
  s.routing_salience=np.square(packed[:,:,8:16].astype(np.float64))*packed[:,:,16:17].astype(np.float64)
  s.routing_hot=packed[:,:,17:25].astype(bool);s.routing_active=packed[:,:,25:33].astype(bool)
  s.ledger.stage(s.positions,s.tokens,packed[:,:,:8].astype(np.int64),
                 packed[:,:,8:16].copy(),packed[:,:,16].copy())
 def sampled(s,num_rejected,num_sampled):
  if s.dummy:return
  rejected=int(num_rejected.detach().cpu().reshape(-1)[0])
  count=len(s.ledger.pending)-rejected
  if count<0:raise RuntimeError('Rejected count exceeds target rows')
  if not s.metric_skip:
   s.metrics.add(s.routing_hot,s.routing_active,count,s.metric_phase,desired=s.routing_desired,salience=s.routing_salience)
   if time.monotonic()-s.metrics.last_write>=2:
    s.metrics.io=s.executor.io_stats("routing_metrics")
    s.metrics.io.update(host_mapped_slots=s.executor.unified,mailbox_pending=len(s.executor.wait_apply))
    s.metrics.io["prefetch"]=s.prefetch.stats() if s.prefetch is not None else {"mode":"legacy"}
    s.metrics.export()
  rows=s.ledger.accept_before(s.ledger.committed_end+count)
  s.pool.wanted=s.predictor.commit(rows)
  if s.prefetch is not None and not s.batch.has_prefill:
   s.prefetch.observe_decode(count,time.monotonic()-s.step_started)
  s.dispatch()
  emitted=int(num_sampled.detach().cpu().reshape(-1)[0])
  if s.tps>0 and not s.batch.has_prefill:s.pace_deadline=s.step_started+emitted/s.tps
RT=Runtime()
