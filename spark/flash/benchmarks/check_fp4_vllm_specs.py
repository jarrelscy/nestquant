"""Check installed vLLM's CPU cache specs via an isolated docker-exec process.

No GPU context or server requests. Sends module source over stdin; neither the
live bind mount nor container files are edited. Requires the server container.
"""

import argparse
import pathlib
import subprocess


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--container", default="glm53-flash-nestquant")
    args = p.parse_args()
    src = (
        pathlib.Path(__file__).resolve().parents[1] / "nq_flash_fp4_kv.py"
    ).read_text()
    script = """import sys, torch, linecache, types
sys.path.insert(0,'/nq/spark/flash')
from nq_flash_attention import install
install()
"""
    script += "src = " + repr(src) + "\n"
    script += """filename='/__nq_fp4_readonly_probe__.py'
linecache.cache[filename]=(len(src), None, src.splitlines(True), filename)
m=types.ModuleType('nq_fp4_probe');m.__file__=filename
sys.modules[m.__name__]=m
exec(compile(src,filename,'exec'),m.__dict__)
m.install()
from vllm.v1.kv_cache_interface import MLAAttentionSpec, get_kv_quant_mode
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import FlashInferMLASparseSM120Backend as Backend
for dtype in ('fp8', 'fp8_ds_mla'):
 s=MLAAttentionSpec(block_size=1,num_kv_heads=1,head_size=512,dtype=torch.uint8,cache_dtype_str=dtype,kv_quant_mode=get_kv_quant_mode(dtype))
 assert s.real_page_size_bytes==320,(dtype,s.real_page_size_bytes)
 s=MLAAttentionSpec(block_size=256,num_kv_heads=1,head_size=512,dtype=torch.uint8,cache_dtype_str=dtype,model_version=m.FORMAT,kv_quant_mode=get_kv_quant_mode(dtype))
 assert s.real_page_size_bytes==81920
 assert MLAAttentionSpec.merge([s,s]).real_page_size_bytes==81920
print('vLLM alignment probe, stamped and merged specs: OK')
from types import SimpleNamespace
from vllm.model_executor.layers.attention.mla_attention import MLAAttention
layer=SimpleNamespace(kv_cache_dtype='fp8_ds_mla',head_size=512,sliding_window=None,
    non_causal_multi_token_decode=False,attn_backend=Backend,prefill_backend=None)
config=SimpleNamespace(cache_config=SimpleNamespace(block_size=256,cache_dtype='fp8_ds_mla'),model_config=None)
spec=MLAAttention.get_kv_cache_spec(layer,config)
assert spec.model_version==m.FORMAT and spec.real_page_size_bytes==81920
layer.prefill_backend=object()
try:
 MLAAttention.get_kv_cache_spec(layer,config)
except ValueError:
 pass
else:
 raise AssertionError('Dense prefill cache reader was not rejected')
print('Actual MLA spec stamping and incompatible prefill rejection: OK')
assert Backend.get_kv_cache_shape(4,256,1,512,'fp8_ds_mla')==(4,256,320)
print('vLLM backend shape: OK')
from vllm.v1.worker.utils import select_common_block_size
assert select_common_block_size(14080,[Backend])==14080
print('Hybrid manager pages remain intact: OK')
s=MLAAttentionSpec(block_size=256,num_kv_heads=1,head_size=576,dtype=torch.uint8,cache_dtype_str='fp8_ds_mla')
assert s.real_page_size_bytes==256*656
print('Other MLA specs unchanged: OK')
from vllm.v1.worker.gpu.attn_utils import _reshape_attention_kv_cache
from dataclasses import replace
s=MLAAttentionSpec(block_size=256,num_kv_heads=1,head_size=512,dtype=torch.uint8,cache_dtype_str='fp8_ds_mla',model_version=m.FORMAT,indexes_kv_by_block_stride=True)
s=replace(s,page_size_padded=s.real_page_size_bytes+256)
buf=torch.zeros(4*s.page_size_bytes,dtype=torch.uint8)
c=_reshape_attention_kv_cache(buf,s,(4,256,320),(0,1,2),4,None)
assert c.stride()==(82176,320,1),c.stride()
assert m._cache_bytes(c).stride()==c.stride()
print('Actual vLLM padded-page reshape: OK',c.shape,c.stride())
assert not torch.cuda.is_initialized()
print('CUDA context never initialized: OK')
"""
    subprocess.run(
        [
            "docker",
            "exec",
            "-i",
            "-e",
            "CUDA_VISIBLE_DEVICES=",
            "-e",
            "NQ_FLASH_ENABLE=0",
            args.container,
            "python3",
            "-",
        ],
        input=script,
        text=True,
        check=True,
    )


if __name__ == "__main__":
    main()
