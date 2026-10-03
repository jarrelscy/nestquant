# bit-exactness of nq_fp8o::gemv_bf16 vs the host-chain fp8_w8a16 path (run in the serve container: /opt/vllm/.venv/bin/python)
import sys,os,torch
sys.path.insert(0,'/opt/vllm');from vllm.model_executor.layers.quantization import fp8_w8a16 as m
from torch.utils.cpp_extension import load
load(name='nq_fp8o',sources=[os.path.join(os.path.dirname(os.path.abspath(__file__)),'nq_fp8o.cu')],extra_cuda_cflags=['-O3'],is_python_module=False)
torch.manual_seed(0);E=m._get_ext()
for (N,K,M) in [(1,4096,6144),(4,4096,6144),(8,2048,6144),(16,16384,1536)]:
    w=torch.randn(M,K,device='cuda');sc=w.abs().amax(1)/448;wq=(w/sc[:,None]).to(torch.float8_e4m3fn).view(torch.uint8).contiguous();sc=sc.float().contiguous()
    for mag in (1.,30.,1e-3):
        x=(torch.randn(N,K,device='cuda')*mag).bfloat16()
        ref=E.fp8_w8a16_gemv((x.float()*2**-6).half().contiguous(),wq,sc).to(torch.bfloat16)
        y=torch.ops.nq_fp8o.gemv_bf16(x,wq,sc)
        print(N,K,M,mag,'bitexact',torch.equal(y.view(torch.int16),ref.view(torch.int16)),'maxdiff',(y.float()-ref.float()).abs().max().item())
x=torch.randn(4,4096,device='cuda').bfloat16();w=torch.randn(6144,4096,device='cuda');sc=w.abs().amax(1)/448;wq=(w/sc[:,None]).to(torch.float8_e4m3fn).view(torch.uint8).contiguous();sc=sc.float().contiguous()
def old():return E.fp8_w8a16_gemv((x.to(torch.float32)*2**-6).half().contiguous(),wq,sc).to(torch.bfloat16)
def new():return torch.ops.nq_fp8o.gemv_bf16(x,wq,sc)
for f in (old,new,old,new):
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(78):f()
    for _ in range(3):g.replay()
    torch.cuda.synchronize();t=torch.cuda.Event(enable_timing=True);u=torch.cuda.Event(enable_timing=True);t.record()
    for _ in range(50):g.replay()
    u.record();torch.cuda.synchronize();print(f.__name__,'us per 78 calls',t.elapsed_time(u)/50*1e3)
