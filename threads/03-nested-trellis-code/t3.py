import torch
torch.cuda.set_per_process_memory_fraction(12/80)
from trellis import *
from exllamav3.modules.quant.exl3_lib.quantize import quantize_tiles
torch.manual_seed(0)
x = torch.randn(64, 256, device='cuda')
x0=x.clone(); q,idx=quantize_tiles(x.contiguous(),{"K":2,"mul1":True})
i=idx.long()&0xffff
ok=(((i[:,:-1]<<2)&0xffff)>>2 == (i[:,1:]>>2)).float().mean()
print('exl3 path valid frac',ok.item(), 'wrap', ((((i[:,-1]<<2)&0xffff)>>2)==(i[:,0]>>2)).float().mean().item())
print("exl3 mse",((x0-q)**2).mean().item(), "x changed", (x0-x).abs().max().item())
C=mul1_codebook(16)[:,None]
qq,ws=viterbi(x.view(64,256,1),C,16,2)
print('mine',((x-qq.view(64,256))**2).mean().item())
print('mine valid', (((ws[:,:-1]<<2)&0xffff)>>2 == (ws[:,1:]>>2)).float().mean().item())
import trellis
# free pass only
B=64;L=16;k=2
def free(x):
    xb=x.view(B,256,1)
    S=2**L;Sp=2**(L-k);nk=4
    cost=torch.zeros(B,S,device='cuda')
    for t in range(256):
        m,a=cost.view(B,nk,Sp).min(dim=1)
        d=((xb[:,t,:][:,None,:]-C[None])**2).sum(-1)
        cost=(m[:,:,None]+d.view(B,Sp,nk)).view(B,S)
    return cost.min(1).values.sum().item()/(B*256)
print('free cost', free(x))
