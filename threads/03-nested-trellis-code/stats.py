import torch
torch.cuda.set_per_process_memory_fraction(12/80)
from exllamav3.modules.quant.exl3_lib.quantize import quantize_tiles
torch.manual_seed(0)
x=torch.randn(2048,256,device='cuda')
q2,_=quantize_tiles(x.contiguous(),{"K":2,"mul1":True}); e=x-q2
v=e.var().item()
print('var',v)
for lag in [1,2,3,4,8]:
    print('acf',lag, (e[:,lag:]*e[:,:-lag]).mean().item()/v)
# conditional variance by q2 bucket and by |x|
qs=torch.quantile(q2.flatten()[:1000000],torch.linspace(0,1,9,device='cuda'))
import math
vs=[];ps=[]
for i in range(8):
    m=(q2>=qs[i])&(q2<=qs[i+1]); vs.append(e[m].var().item()); ps.append(m.float().mean().item())
    print(f'q2 bucket {qs[i].item():+.2f}..{qs[i+1].item():+.2f} var {vs[-1]:.5f} mean {e[m].mean().item():+.4f}')
am=sum(p*v for p,v in zip(ps,vs)); gm=math.exp(sum(p*math.log(v) for p,v in zip(ps,vs)))
print('AM/GM dB',10*math.log10(am/gm))
print('per-tile var spread', e.var(1).std().item()/v)
