"""Run existing metadata-free MLA startup path before hot-pool pinning."""
import hashlib,json,pathlib,time

def select_attention(model):
 matches=[m for n,m in model.named_modules() if n=='language_model.model.layers.7']
 if len(matches)!=1:raise RuntimeError('Expected exact target layer7')
 layer=matches[0]
 if layer.layer_kind!='mla' or type(layer.self_attn).__name__!='Glm5NextMLAAttention':raise RuntimeError('Wrong startup MLA layer')
 return layer.self_attn

def run(runner,x):
 import torch
 from vllm.forward_context import get_forward_context
 if get_forward_context().attn_metadata is not None:raise RuntimeError('Prewarm only allowed without attention metadata')
 if tuple(x.shape)!=(1024,4096) or x.dtype!=torch.bfloat16 or not x.is_cuda:raise RuntimeError('Unexpected unchanged1024 profile shape')
 if getattr(runner,'_nq_mla_prewarmed',False):raise RuntimeError('MLA prewarm repeated')
 attn=select_attention(runner.model)
 p=pathlib.Path('/data/kld-sampled-ahead-sentinel');t=time.monotonic();started_at=time.time()
 from vllm.v1.worker.workspace import current_workspace_manager
 def workspace():
  return [None if t is None else {'bytes':t.numel()*t.element_size(),'ptr':t.data_ptr()} for t in current_workspace_manager()._current_workspaces]
 workspace_before=workspace()
 before={'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved()}
 torch.cuda.synchronize()
 hidden=torch.zeros_like(x);positions=torch.arange(1024,device=x.device,dtype=torch.int64)
 with torch.inference_mode():out=attn(hidden,positions)
 if tuple(out.shape)!=tuple(x.shape) or out.dtype!=x.dtype:raise RuntimeError('Wrong MLA profile output')
 torch.cuda.synchronize()
 del out,hidden,positions
 runner._nq_mla_prewarmed=True
 receipt={'started_at':started_at,'completed_at':time.time(),'phase':'before_hot_pool_after_KV_reservation','tokens':1024,'hidden':4096,'dtype':str(x.dtype),'metadata':None,'elapsed_seconds':time.monotonic()-t,'before':before,'after':{'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved()},'workspace_before':workspace_before,'workspace_after':workspace(),'retained_workspace':'existing global workspace manager; no reset','real_profile_retained':True,'no_expert_or_KDA_execution':True}
 (p/'mla-prewarm.json').write_text(json.dumps(receipt,indent=2));print(json.dumps({'nq_mla_prewarm':receipt}),flush=True)
