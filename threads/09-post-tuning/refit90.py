"""Refit EXL3 (same upstream call as orbit-duet benchmarks.fit_matched_exl3) on Grams from the 90% tuning split only,
so the 10% held-out split is out-of-sample for BOTH the codes and the post-tuning. Output: SCRATCH/exl3_90_{model}/expert_{bits}.bin"""
import sys, os, time, json
import torch
import torch.nn.functional as F
from common import *
from orbit_duet.exl3_adapter import write_legacy
from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3, get_temp_buffers
setup()
model = sys.argv[1]
c = CFG[model]
w = teacher_weights(model)
s = torch.load(c['sample'], mmap=True, weights_only=True)
N = len(s['p']); g = torch.Generator().manual_seed(90909); perm = torch.randperm(N, generator=g)
train = perm[N // 10:].sort().values
grams = [torch.zeros(6144, 6144, device='cuda'), torch.zeros(2048, 2048, device='cuda')]
with torch.no_grad():
    for st in range(0, len(train), 512):
        ids = train[st:st + 512]
        x = s['x'][ids].cuda(); p = s['p'][ids].cuda().float()[:, None]
        h = (F.silu(F.linear(x.bfloat16(), w[0].bfloat16())) * F.linear(x.bfloat16(), w[1].bfloat16())).float()
        for dst, v in zip(grams, [x.float() * p, h * p]): dst.addmm_(v.T, v)
out = f'{SCRATCH}/exl3_90_{model}'; os.makedirs(out, exist_ok=True)
for bits in [2, 4]:
    values = []
    for i, weight in enumerate(w):
        qa = dict(K=bits, devices=['cuda:0'], seed=91426, sigma_reg=.03, apply_out_scales=None, mul1=True)
        hdata = dict(H=grams[int(i == 2)].clone(), count=len(train), finalized=False, device=torch.device('cuda:0'))
        with torch.no_grad(): _, proxy, value = quantize_exl3(weight.T.contiguous(), hdata, qa, False, verbose=False)
        values.append(dict(shape=list(weight.shape), **{k: v.cpu() for k, v in value.items()}))
        get_temp_buffers.cache_clear(); del hdata, value; torch.cuda.empty_cache()
        print(bits, i, proxy, flush=True)
    write_legacy(f'{out}/expert_{bits}.bin', values)
print('done')
