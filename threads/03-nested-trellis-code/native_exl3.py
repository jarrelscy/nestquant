import torch, json, time
torch.cuda.set_per_process_memory_fraction(12/80)
from exllamav3.modules.quant.exl3_lib.quantize import quantize_tiles
torch.manual_seed(0)
N=2048
x=torch.randn(N,256,device='cuda')
res={}
for cbname in ['mul1','mcg','3inst']:
  for K in [2,3,4]:
    best=None
    for s in [0.8,0.9,1.0,1.1,1.2,1.3]:
        qa={"K":K}
        if cbname!='3inst': qa[cbname]=True
        t=time.time()
        q,idx=quantize_tiles((x*s).contiguous(),qa); torch.cuda.synchronize()
        mse=((x-q/s)**2).mean().item()
        if best is None or mse<best[0]: best=(mse,s,time.time()-t)
    res[f"{cbname}_K{K}"]=best; print(cbname,K,best,flush=True)
json.dump(res,open('native_exl3.json','w'),indent=1)
