"""Opt-in integration with vLLM 487ecf187 GLM5Next and ModelRunnerV2."""
import functools,inspect,json,os,re


def install():
 if os.environ.get('NQ_FLASH_ENABLE')!='1':return
 import torch
 from nq_flash_attention import install as install_attention
 install_attention()
 kv_format=os.environ.get("NQ_FLASH_MLA_CACHE", "fp8")
 if kv_format not in ("fp8", "fp4_g16"):
  raise ValueError("NQ_FLASH_MLA_CACHE must be fp8 or fp4_g16")
 if kv_format=="fp4_g16":
  from nq_flash_fp4_kv import install as install_fp4_kv
  install_fp4_kv()
 # SM12x DeepGEMM FP8 indexer supports 64-entry pages, not 32.
 # This also makes kpool storage block alignment 4*64 tokens.
 import vllm.utils.deep_gemm as deep_gemm
 import vllm.v1.worker.utils as worker_utils
 deep_gemm.PAGED_MQA_PAGE_SIZES = (64,)
 worker_utils.PAGED_MQA_PAGE_SIZES = (64,)
 from vllm.config import get_current_vllm_config
 from vllm.distributed import get_tensor_model_parallel_world_size
 from vllm.model_executor.layers.quantization.fp8 import Fp8Config,Fp8LinearMethod
 from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
 from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
 from vllm.model_executor.layers.fused_moe.fused_moe_method_base import FusedMoEMethodBase
 from vllm.models.glm5next.nvidia.model import Glm5NextModel
 from vllm.v1.worker.gpu.model_runner import GPUModelRunner
 from nq_flash_runtime import RT
 if getattr(Fp8Config,'_nq_flash_installed',False):return
 kinds=json.load(open(os.environ['NQ_FLASH_MODEL']+'/config.json'))['text_config']['layer_types']
 class FlashMoEMethod(FusedMoEMethodBase):
  def __init__(s,layer,L):
   super().__init__(layer.moe_config);s.L=L
   cfg=get_current_vllm_config()
   if get_tensor_model_parallel_world_size()!=1 or cfg.parallel_config.enable_expert_parallel or cfg.parallel_config.enable_eplb:
    raise ValueError('Flash streaming currently requires TP1, no EP/EPLB')
   if not cfg.model_config.enforce_eager:raise ValueError('Flash committed routing capture requires --enforce-eager')
   if cfg.cache_config.enable_prefix_caching:raise ValueError('Flash prefix predictor-history restore is not implemented')
   if cfg.scheduler_config.max_num_seqs!=1:raise ValueError('Flash streaming requires --max-num-seqs 1')
   if cfg.speculative_config is not None and cfg.speculative_config.num_speculative_tokens not in (1,2):
    raise ValueError('Flash MTP port currently supports one or two drafts')
  def create_weights(s,layer,num_experts,hidden_size,intermediate_size_per_partition,params_dtype,**attrs):
   if (num_experts,hidden_size,intermediate_size_per_partition)!=(288,4096,2048):raise ValueError('Wrong Flash expert dimensions')
  def process_weights_after_loading(s,layer):RT.add_layer(s.L,torch.device('cuda',torch.cuda.current_device()))
  def get_fused_moe_quant_config(s,layer):return None
  def apply(s,layer,x,topk_weights,topk_ids,shared_experts=None,shared_experts_input=None):
   return RT.forward(s.L,x,topk_weights,topk_ids)
 class FlashFp8LMHead(Fp8LinearMethod):
  def create_weights(self,layer,*a,**kw):
   super().create_weights(layer,*a,**kw)
   def load_scale(param,value,*args,**kwargs):
    if param.shape!=value.shape:raise ValueError(f'LM head block-scale shape {value.shape} != {param.shape}')
    param.data.copy_(value)
   # VocabParallelEmbedding's loader shards vocabulary rows, not scale blocks.
   layer.weight_scale_inv.weight_loader=load_scale
 original=Fp8Config.get_quant_method
 @functools.wraps(original)
 def get_quant_method(self,layer,prefix):
  m=re.search(r'(?:^|\.)layers\.(\d+)\.mlp\.experts$',prefix)
  if isinstance(layer,RoutedExperts) and m and 3<=int(m[1])<45:return FlashMoEMethod(layer,int(m[1]))
  # Only the converted targets bypass the checkpoint's BF16 exclusion list.
  a=re.search(r'layers\.(\d+)\.self_attn\.(\w+)$',prefix)
  converted=isinstance(layer,ParallelLMHead)
  if a:
   L,p=int(a[1]),a[2];kda=L<len(kinds) and kinds[L]=='linear_attention'
   converted=(kda and p in ('in_proj_qkv','o_proj')) or (not kda and p=='kv_b_proj')
  if converted and os.environ.get('NQ_FLASH_NATIVE_FP8')=='1':
   return FlashFp8LMHead(self) if isinstance(layer,ParallelLMHead) else Fp8LinearMethod(self)
  return original(self,layer,prefix)
 Fp8Config.get_quant_method=get_quant_method
 original_forward=Glm5NextModel.forward
 @functools.wraps(original_forward)
 def forward(self,input_ids,positions,intermediate_tensors,inputs_embeds=None,**kwargs):
  RT.begin(input_ids,positions)
  result=original_forward(self,input_ids,positions,intermediate_tensors,inputs_embeds,**kwargs)
  RT.finish();return result
 Glm5NextModel.forward=forward
 original_execute=GPUModelRunner.execute_model;sig=inspect.signature(original_execute)
 @functools.wraps(original_execute)
 def execute(self,*a,**kw):
  args=sig.bind(self,*a,**kw);args.apply_defaults();RT.dummy=args.arguments['dummy_run']
  return original_execute(self,*a,**kw)
 GPUModelRunner.execute_model=execute
 original_prepare=GPUModelRunner.prepare_inputs
 @functools.wraps(original_prepare)
 def prepare(self,*a,**kw):
  batch=original_prepare(self,*a,**kw);RT.batch=batch;return batch
 GPUModelRunner.prepare_inputs=prepare
 original_sample=GPUModelRunner.sample
 @functools.wraps(original_sample)
 def sample(self,*a,**kw):
  output=original_sample(self,*a,**kw)
  RT.sampled(num_rejected=output[2],num_sampled=output[1]);return output
 GPUModelRunner.sample=sample
 Fp8Config._nq_flash_installed=True
