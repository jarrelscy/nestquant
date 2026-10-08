"""Check NoPE/DSA+SWA adapter against dense attention on identical decoded FP8 KV."""
import json, math
from types import SimpleNamespace
import torch
from nq_flash_attention import install
install()
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm120 import FlashInferMLASparseSM120Impl as Impl

torch.manual_seed(43)
dev='cuda';h=64;length=3072
values=torch.randn(length,512,device=dev).to(torch.bfloat16)
cache=torch.zeros(length,656,device=dev,dtype=torch.uint8)
scales=torch.rand(length,4,device=dev)*.02+.005
encoded=(values.float().reshape(length,4,128)/scales[...,None]).clamp(-448,448).to(torch.float8_e4m3fn)
cache[:,:512]=encoded.reshape(length,512).view(torch.uint8)
cache[:,512:528]=scales.contiguous().view(torch.uint8)
decoded=(encoded.float()*scales[...,None]).reshape(length,512)
for n in (1,4,65):
 for empty_main in (False,True):
  impl=Impl.__new__(Impl)
  impl._nq_nope=True;impl.num_heads=h;impl.kv_lora_rank=512;impl.qk_rope_head_dim=64;impl.qk_nope_head_dim=128
  impl.scale=1/math.sqrt(128);impl.kv_scale_format='arbitrary_fp32';impl._workspace_buffer=None
  idx=torch.arange(2176,device=dev,dtype=torch.int32).repeat(n,1)
  if empty_main:idx[:,:2048]=-1
  impl.topk_indices_buffer=idx
  metadata=SimpleNamespace(req_id_per_token=torch.zeros(n,device=dev,dtype=torch.int32),block_table=torch.arange(length//128,device=dev,dtype=torch.int32)[None],block_size=128,topk_tokens=2048)
  q=torch.randn(n,h,512,device=dev,dtype=torch.bfloat16)*.3
  out,_=impl.forward_mqa(q,cache.view(-1,128,656),metadata,None)
  keys=decoded[idx.clamp_min(0).long()]
  logits=torch.einsum('nhd,nkd->nhk',q.float(),keys)*impl.scale
  logits.masked_fill_(idx[:,None,:]<0,float('-inf'))
  ref=torch.einsum('nhk,nkd->nhd',logits.softmax(-1),keys)
  diff=out.float()-ref;rel=float(diff.norm()/ref.norm());ma=float(diff.abs().max())
  print(json.dumps(dict(tokens=n,empty_main=empty_main,max_abs=ma,relative_l2=rel)),flush=True)
  assert torch.isfinite(out).all() and rel<.04,(rel,ma)
print('ATTENTION_PARITY_PASS',flush=True)
