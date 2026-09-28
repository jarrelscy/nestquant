import torch
torch.cuda.set_per_process_memory_fraction(12/80)
from trellis import *
from exllamav3.modules.quant.exl3_lib.quantize import quantize_tiles
torch.manual_seed(0)
x = torch.randn(4, 256, device='cuda')
q,idx=quantize_tiles(x.contiguous(),{"K":2,"mul1":True})
C=mul1_codebook(16)
i=idx.long()&0xffff
print((C[i]-q).abs().max().item())
print(i[0,:12].tolist())
print([bin(v)[2:].zfill(16) for v in i[0,:6].tolist()])
