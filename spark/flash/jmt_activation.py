"""Experimental calibrated activation/first-use adapter; committed inputs only."""
import sys,os,json,hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from jt_runtime import IncrementalJT,module
from jmt_heads import BatchedHeads

BENEFIT_FIELD='measured_local_teacher_error_reduction'
def load_benefits(root,model_root=None):
 root=Path(root);manifest=json.loads((root/'RELEASE_COMPLETE.json').read_text())
 if not manifest.get('complete') or manifest.get('scope')!='Flash 1.5-to-4 expert upgrade weighting metadata only':raise ValueError('Wrong benefit manifest')
 for n,e in manifest['files'].items():
  p=Path(n)
  if p.is_absolute() or '..' in p.parts:raise ValueError('Unsafe benefit path')
  b=(root/n).read_bytes()
  if len(b)!=e['bytes'] or hashlib.sha256(b).hexdigest()!=e['sha256']:raise ValueError('Benefit integrity mismatch '+n)
 contract=json.loads((root/'CONTRACT.json').read_text())
 if contract['precision_pairs']!={'flash':[1.5,4]} or contract['layers']!=42 or contract['experts']!=12096:raise ValueError('Wrong benefit dimensions/precision')
 values=np.zeros((42,288));sal=np.zeros_like(values);support=[];negative=0
 for i,L in enumerate(range(3,45)):
  d=json.loads((root/f'L{L}.json').read_text())
  if (d['model'],d['layer'],d['base_bits'],d['upgrade_level'])!=('flash',L,1.5,4):raise ValueError('Wrong layer metadata')
  if model_root and hashlib.sha256((Path(model_root)/f'layers/L{L}/manifest.json').read_bytes()).hexdigest()!=d['manifest_sha256']:raise ValueError('Checkpoint manifest mismatch')
  if len(d['experts'])!=288 or set(e['expert'] for e in d['experts'])!=set(range(288)):raise ValueError('Wrong expert IDs')
  for e in d['experts']:
   j=e['expert'];v=e[BENEFIT_FIELD];x=v['mean']
   if not np.isfinite(x) or v['sample_count']<=0:raise ValueError('Missing benefit sample')
   negative+=int(x<0);values[i,j]=max(0.,x)
   sal[i,j]=e['conditional_normalized_squared_salience']['mean']
   support.append(v['sample_count'])
 return values,dict(field=BENEFIT_FIELD,negative_clipped=negative,min_samples=min(support),salience_already_included=True,not_lm_loss=True),sal

class ActivationHeads(nn.Module):
 def __init__(self,projection,down,up,bias,calibration,layers=42,experts=288):
  super().__init__();self.batched=BatchedHeads(projection,down,up,bias)
  self.out_features=projection.out_features;self.layers=layers;self.experts=experts
  self.register_buffer('slope',calibration['slope']);self.register_buffer('intercept',calibration['intercept'])
 def forward(self,x):
  z=self.batched(x,1,31).view(31,*x.shape[:-1],self.layers,self.experts)
  cumulative=torch.cat((z[:1],z[:1]+torch.nn.functional.softplus(z[16:]).cumsum(0)),0)
  raw=torch.cat((z[:16],cumulative),0)
  shape=(32,)+(1,)*(x.ndim-1)+(self.layers,self.experts)
  return torch.sigmoid(raw.float()*self.slope.reshape(shape)+self.intercept.reshape(shape))

class ProbabilityPool(nn.Module):
 def __init__(self,heads,benefit):
  super().__init__();self.heads=heads;self.out_features=heads.out_features
  self.register_buffer('benefit',torch.as_tensor(benefit,dtype=torch.float32,device=heads.slope.device))
  self.forecasts=None
 def forward(self,x):
  # Only the most recent committed position drives the joint planner. The
  # legacy initialization policy consumes a normalized value proxy, NOT p/F.
  probability=self.heads(x[:,-1:])[:,0,0]
  pair=probability.detach().cpu().numpy()
  self.forecasts={'activation':pair[:16], 'first_use':pair[16:]}
  value=probability[:16].mean(0)*self.benefit
  log=value.clamp_min(1e-30).log().flatten()
  return log[None,None].expand(*x.shape[:-1],self.out_features)

class ActivationIncremental(IncrementalJT):
 def __init__(self,root,device,dtype):
  if dtype!=torch.float32:raise ValueError('Activation jMT requires validated float32')
  root=Path(root)
  # Published model uses an absolute sibling import; isolate its module name.
  old=sys.modules.get('jt_model');api_jt=module(root/'jt_model.py','activation_release_jt')
  sys.modules['jt_model']=api_jt
  try:
   api=module(root/'model.py','activation_release_model')
   from safetensors.torch import load_file
   cfg=json.loads((root/'jMT_activation.json').read_text())
   # vLLM can set a global half dtype/default CUDA device. Convert the empty
   # module BEFORE loading FP32 weights, avoiding irreversible half rounding.
   with torch.device('cpu'):
    m=api.ActivationJMT(layers=cfg['layers'],experts=cfg['experts'],**cfg['base_config']).float()
    m.load_state_dict(load_file(str(root/'jMT_activation.safetensors')))
   m=m.to(device).eval();cal=load_file(str(root/'calibration.safetensors'),device=str(device))
  finally:
   if old is None:sys.modules.pop('jt_model',None)
   else:sys.modules['jt_model']=old
  objective=os.environ.get('NQ_FLASH_JMT_OBJECTIVE','benefit')
  if objective not in ('benefit','hits'):raise ValueError('Unknown probability objective')
  if objective=='hits':
   benefit=np.ones((42,288),dtype=np.float32);self.benefit_info=dict(field='unit_value_per_activation',objective='hits',salience_weighting=False,upgrade_weighting=False)
  else:benefit,self.benefit_info,_=load_benefits(os.environ['NQ_FLASH_JMT_BENEFIT'],os.environ['NQ_FLASH_MODEL'])
  heads=ActivationHeads(m.bhead,m.down,m.up,m.offset_bias,cal)
  self.pool=ProbabilityPool(heads,benefit);m.bhead=self.pool
  super().__init__(m,api_jt.rope)
 def reset(self):
  super().reset();self.pool.forecasts=None
 def stats(self):
  return dict(parameter_dtype=str(next(self.model.parameters()).dtype),mode='activation_firstuse',horizons=[1,16],routing_ema_mix=0.,benefit=self.benefit_info)
