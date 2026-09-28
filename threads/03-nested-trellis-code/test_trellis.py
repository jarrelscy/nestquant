import torch, time
torch.cuda.set_per_process_memory_fraction(12/80)
from trellis import *
torch.manual_seed(0)
x = torch.randn(512, 256, device='cuda')
for L,K in [(16,2),(16,4),(16,1)]:
    C = mul1_codebook(L)[:, None]
    s = {2:1.0,4:0.925,1:1.0}[K]
    t=time.time(); q, w = fit(x*s, C, L, K); torch.cuda.synchronize()
    print(L,K,((x - q/s)**2).mean().item(), time.time()-t)
from exllamav3.modules.quant.exl3_lib.quantize import quantize_tiles
for K,s in [(2,1.0),(4,0.925),(1,1.0)]:
    q,_=quantize_tiles((x*s).contiguous(),{"K":K,"mul1":True}); print('exl3',K,((x-q/s)**2).mean().item())
