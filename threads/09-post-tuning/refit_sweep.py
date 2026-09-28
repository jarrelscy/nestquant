"""EXL3 refits (upstream quantize_exl3, same call as fit_matched_exl3) varying K and sigma_reg.
split=full: Grams from the saved full statistics (as the matched artifacts); split=90: Grams from the 90% tuning split.
Reports 10%-held-out p2-weighted rel L2 (clean only for split=90) and evaluation-capture errors.
usage: refit_sweep.py MODEL SPLIT K:SIGMA [K:SIGMA ...]"""
import sys, os, json, torch
import torch.nn.functional as F
from common import *
from orbit_duet.exl3_adapter import write_legacy, EXL3Expert
from orbit_duet.statistics import load_statistics
from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3, get_temp_buffers
setup()
model, split = sys.argv[1], sys.argv[2]
c = CFG[model]; w = teacher_weights(model)
s = torch.load(c['sample'], mmap=True, weights_only=True)
N = len(s['p']); g = torch.Generator().manual_seed(90909); perm = torch.randperm(N, generator=g)
hold, train = perm[:N // 10].sort().values, perm[N // 10:].sort().values
if split == 'full':
    st = load_statistics(c['sample'].replace('_training_sample.pt', '.pt'), w, c['L'], c['E'])
    grams = [v.cuda().clone() for v in st['grams']]; count = st['metadata']['training_rows']
else:
    grams = [torch.zeros(6144, 6144, device='cuda'), torch.zeros(2048, 2048, device='cuda')]; count = len(train)
    with torch.no_grad():
        for i in range(0, len(train), 512):
            ids = train[i:i + 512]; x = s['x'][ids].cuda(); p = s['p'][ids].cuda().float()[:, None]
            h = (F.silu(F.linear(x.bfloat16(), w[0].bfloat16())) * F.linear(x.bfloat16(), w[1].bfloat16())).float()
            for dst, v in zip(grams, [x.float() * p, h * p]): dst.addmm_(v.T, v)
Xh = s['x'][hold].cuda(); Ph = s['p'][hold].float().cuda().square()
with torch.no_grad(): Yh = expert(Xh, w)
def hold_err(W):
    with torch.no_grad():
        e = ((expert(Xh, W) - Yh).square().sum(-1) * Ph).sum() / (Yh.square().sum(-1) * Ph).sum()
    return 100 * float(e) ** .5
out = f'{SCRATCH}/exl3_sweep_{model}_{split}'; os.makedirs(out, exist_ok=True)
res = {}
for spec in sys.argv[3:]:
    K, sig = spec.split(':'); K = float(K) if '.' in K else int(K); sig = float(sig)
    values = []
    for i, weight in enumerate(w):
        qa = dict(K=K, devices=['cuda:0'], seed=91426, sigma_reg=sig, apply_out_scales=None, mul1=True)
        hdata = dict(H=grams[int(i == 2)].clone(), count=count, finalized=False, device=torch.device('cuda:0'))
        with torch.no_grad(): _, proxy, value = quantize_exl3(weight.T.contiguous(), hdata, qa, False, verbose=False)
        values.append(dict(shape=list(weight.shape), **{k: v.cpu() for k, v in value.items()}))
        get_temp_buffers.cache_clear(); del hdata, value; torch.cuda.empty_cache()
    path = f'{out}/expert_{K}_{sig}.bin'; write_legacy(path, values)
    ref = EXL3Expert(path).decoded_weights()
    ev = eval_captures(model, w, {'q': (ref, [None] * 3)})
    r = dict(K=K, sigma_reg=sig, bpw=os.path.getsize(path) * 8 / 37748736, hold=hold_err(ref), eval={k: v['q'] for k, v in ev.items()})
    res[spec] = r
    print(json.dumps(r), flush=True)
    if not os.environ.get('KEEP'): os.remove(path)
os.makedirs('results_refit', exist_ok=True)
json.dump(res, open(f'results_refit/{model}_{split}_{"_".join(a.replace(":", "s") for a in sys.argv[3:])}.json', 'w'), indent=1)
