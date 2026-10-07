# nq-kld debug-only numerics switches for the MTP verify path (m=4 rows) vs plain decode (m=1). Off unless the env is set.
#   NQ_DBG_FUSE_M=<n>   b12x fused AR+add+RMSNorm (fp32 one-shot) also on static compile ranges [k,k], k<=n (prod: only [1,1];
#                       needs compile_sizes to include k and VLLM_PCIE_ONESHOT_FUSED_ADD_RMS_NORM_MAX_SIZE >= k*12288)
#   NQ_DBG_LIN_ROWSPLIT=<n> unquantized (bf16) linears incl. lm_head: inputs with 2<=rows<=n run as n separate m=1 GEMMs
#                       (opaque custom op nq_dbg::rowsplit_linear, so it survives torch.compile/cudagraphs)
#   NQ_DBG_MLA_BMM_ROWSPLIT=<n> MLA absorb bmms (q_nope@W_UK_T, attn_out@W_UV; (N,B,*) layout) with 2<=B<=n as B single-row bmms
#   NQ_DBG_NO_BF16_RED=1 torch.backends.cuda.matmul.allow_{bf16,fp16}_reduced_precision_reduction=(False,True): no reduced-precision (bf16) split-K reductions; split-K itself allowed (allow_splitk=False needs cuBLASLt)
#   NQ_DBG_LMHEAD_FP32=1 lm_head GEMM with fp32 output (torch.mm out_dtype=float32) for every row count
# Installed from sitecustomize as post-import hooks (no early vllm import).
import os,sys,logging
log=logging.getLogger('nq_dbg_numerics')
FUSE_M=int(os.environ.get('NQ_DBG_FUSE_M','0') or 0)

def patch_fuse(m):
    C=m.B12xAllReduceRMSFusionPass
    def is_applicable_for_range(self,r):
        return (r.start==r.end and 1<=r.start<=FUSE_M and self.hidden_size==6144 and self.max_bytes>=12288*r.start)
    C.is_applicable_for_range=is_applicable_for_range
    print(f'nq_dbg_numerics: b12x AR+RMS fusion on static ranges [k,k] k<={FUSE_M}',file=sys.stderr)

LIN_RS=int(os.environ.get('NQ_DBG_LIN_ROWSPLIT','0') or 0)

def patch_lin(m):
    import torch
    @torch.library.custom_op('nq_dbg::rowsplit_linear',mutates_args=())
    def rowsplit_linear(x:torch.Tensor,weight:torch.Tensor,bias:torch.Tensor|None=None)->torch.Tensor:
        x2=x.reshape(-1,x.shape[-1])
        if 2<=x2.shape[0]<=LIN_RS:
            out=torch.cat([torch.nn.functional.linear(x2[i:i+1],weight,bias) for i in range(x2.shape[0])],0)
        else:
            out=torch.nn.functional.linear(x2,weight,bias)
        return out.reshape(*x.shape[:-1],weight.shape[0])
    @rowsplit_linear.register_fake
    def _(x,weight,bias=None):
        return x.new_empty((*x.shape[:-1],weight.shape[0]))
    _orig=m.default_unquantized_gemm
    ONLY=os.environ.get('NQ_DBG_LIN_ONLY','')   # '', 'lmhead' or 'nolmhead'
    def default_unquantized_gemm(layer,x,weight,bias=None):
        if ONLY:
            ish=type(layer).__name__=='ParallelLMHead'
            if (ONLY=='lmhead')!=ish:return _orig(layer,x,weight,bias)
        return torch.ops.nq_dbg.rowsplit_linear(x,weight,bias)
    m.default_unquantized_gemm=default_unquantized_gemm
    print(f'nq_dbg_numerics: unquantized linears row-split for 2<=rows<={LIN_RS} only={os.environ.get("NQ_DBG_LIN_ONLY","")!r}',file=sys.stderr)

MLA_RS=int(os.environ.get('NQ_DBG_MLA_BMM_ROWSPLIT','0') or 0)

def patch_mla(m):
    import torch as _t
    class _TorchProxy:
        def __getattr__(self,k):return getattr(_t,k)
        @staticmethod
        def bmm(a,b,out=None):
            B=a.shape[1]
            if not (2<=B<=MLA_RS):return _t.bmm(a,b,out=out)
            res=_t.cat([_t.bmm(a[:,i:i+1],b) for i in range(B)],1)
            if out is None:return res
            out.copy_(res);return out
    m.torch=_TorchProxy()
    print(f'nq_dbg_numerics: MLA absorb bmms row-split for 2<=B<={MLA_RS}',file=sys.stderr)

NO_RED=os.environ.get('NQ_DBG_NO_BF16_RED','0')=='1'

def patch_nored(m):
    import torch
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=(False,True)
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=(False,True)
    print(f'nq_dbg_numerics: reduced-precision reduction off: bf16={torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}',file=sys.stderr)

LMH32=os.environ.get('NQ_DBG_LMHEAD_FP32','0')=='1'

def patch_lmh32(m):
    import torch
    _orig=m.default_unquantized_gemm
    def default_unquantized_gemm(layer,x,weight,bias=None):
        if type(layer).__name__!='ParallelLMHead':return _orig(layer,x,weight,bias)
        x2=x.reshape(-1,x.shape[-1])
        out=torch.mm(x2,weight.t(),out_dtype=torch.float32)
        if bias is not None:out=out+bias.float()
        return out.reshape(*x.shape[:-1],weight.shape[0])
    m.default_unquantized_gemm=default_unquantized_gemm
    print('nq_dbg_numerics: lm_head fp32-output GEMM',file=sys.stderr)

FP8_OPROJ=os.environ.get('NQ_DBG_FP8_OPROJ_ONLY','0')=='1'   # fp8_w8a16 (e4m3 per-out-channel W8A16) restricted to self_attn.o_proj (needs VLLM_DISABLE_FP8_W8A16=0, VLLM_ENABLE_NVFP4_P4_O_PROJ=0)

# NQ_DBG_FP8O_FUSED (default 1): decode gemv = csrc/nq_fp8o.cu (BF16 in/out, FP32 accumulation).
FUSED=os.environ.get('NQ_DBG_FP8O_FUSED','1')=='1'
_FUSED_OK=[False]
def _load_fused():
    if _FUSED_OK[0]:return
    from torch.utils.cpp_extension import load
    load(name='nq_fp8o',sources=[os.path.join(os.path.dirname(os.path.abspath(__file__)),'csrc','nq_fp8o.cu')],extra_cuda_cflags=['-O3'],is_python_module=False,verbose=False)
    _FUSED_OK[0]=True
def patch_fp8o(m):
    m.TARGET_SUFFIXES=("self_attn.o_proj",)
    import torch
    # apply() calls a cpp_extension.load'ed pybind gemv that dynamo cannot trace -> wrap the whole apply math in an opaque custom op
    @torch.library.custom_op('nq_dbg::fp8_w8a16_linear',mutates_args=())
    def fp8_w8a16_linear(x2d:torch.Tensor,weight:torch.Tensor,scale:torch.Tensor,dec_max:int,shift:int)->torch.Tensor:
        Mo=weight.shape[0]
        if x2d.shape[0]<=dec_max or torch.cuda.is_current_stream_capturing():
            if FUSED and x2d.dtype==torch.bfloat16:return torch.ops.nq_fp8o.gemv_bf16(x2d.contiguous(),weight,scale)
            # The legacy extension narrows BF16 inputs, partials and output through
            # half. Keep the fallback safe too (slower diagnostic/non-BF16 path).
            y=torch.nn.functional.linear(x2d.float(),weight.view(torch.float8_e4m3fn).float())*scale
        else:
            xa=x2d.to(torch.float32);xs=xa.abs().amax(dim=1,keepdim=True).clamp_min(1e-8)/448.0
            xq=(xa/xs).clamp(-448,448).to(torch.float8_e4m3fn)
            y=torch._scaled_mm(xq,weight.view(torch.float8_e4m3fn).t(),scale_a=xs.float().contiguous(),scale_b=scale.view(1,Mo).contiguous(),out_dtype=torch.bfloat16)
        return y.to(x2d.dtype)
    @fp8_w8a16_linear.register_fake
    def _(x2d,weight,scale,dec_max,shift):return x2d.new_empty((x2d.shape[0],weight.shape[0]))
    C=m.Fp8W8A16LinearMethod;_pw=C.process_weights_after_loading
    def process_weights_after_loading(self,layer):
        _pw(self,layer)
        if FUSED:_load_fused()
    def apply(self,layer,x,bias=None):
        y=torch.ops.nq_dbg.fp8_w8a16_linear(x.reshape(-1,x.shape[-1]),layer.weight,layer.weight_scale,self.DECODE_MAX_TOKENS,self.ACT_SHIFT)
        if bias is not None:y=y+bias
        return y.view(*x.shape[:-1],layer.weight.shape[0])
    C.process_weights_after_loading=process_weights_after_loading;C.apply=apply
    print(f'nq_dbg_numerics: fp8_w8a16 TARGET_SUFFIXES={m.TARGET_SUFFIXES}',file=sys.stderr)

_TARGETS={}
if MLA_RS>1:_TARGETS['vllm.model_executor.layers.attention.mla_attention']=patch_mla
if LIN_RS>1:_TARGETS['vllm.model_executor.layers.utils']=patch_lin
if FUSE_M>1:_TARGETS['vllm.compilation.passes.fusion.b12x_allreduce_rms']=patch_fuse
if LMH32:_TARGETS['vllm.model_executor.layers.utils']=patch_lmh32
if FP8_OPROJ:_TARGETS['vllm.model_executor.layers.quantization.fp8_w8a16']=patch_fp8o
if NO_RED:_TARGETS['vllm.model_executor.layers.utils']=(lambda m,_p=_TARGETS.get('vllm.model_executor.layers.utils'):(_p and _p(m),patch_nored(m)))

class _Finder:
    def find_spec(self,name,path=None,target=None):
        if name not in _TARGETS:return None
        for f in sys.meta_path:
            if f is self or not hasattr(f,'find_spec'):continue
            spec=f.find_spec(name,path,target)
            if spec is not None:break
        else:return None
        ld=spec.loader;ex0=ld.exec_module
        def exec_module(m,_ex0=ex0,_n=name):
            _ex0(m)
            try:_TARGETS[_n](m)
            except Exception as e:print(f'nq_dbg_numerics: patch of {_n} failed: {e!r}',file=sys.stderr)
        try:ld.exec_module=exec_module
        except Exception:return None
        return spec

def install():
    if not _TARGETS:return
    for n,f in _TARGETS.items():
        if n in sys.modules:f(sys.modules[n])
    if not any(isinstance(f,_Finder) for f in sys.meta_path):sys.meta_path.insert(0,_Finder())
