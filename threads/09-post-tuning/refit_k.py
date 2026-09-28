"""EXL3 refits at intermediate K on the SAME full saved statistics as the matched artifacts (rate-distortion slope reference)."""
import sys, os, torch
from common import *
from orbit_duet.exl3_adapter import write_legacy
from orbit_duet.statistics import load_statistics
from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3, get_temp_buffers
setup()
model = sys.argv[1]; Ks = [float(k) if '.' in k else int(k) for k in sys.argv[2:]]
c = CFG[model]; w = teacher_weights(model)
statp = c['sample'].replace('_training_sample.pt', '.pt')
st = load_statistics(statp, w, c['L'], c['E'])
grams = [v.cuda().clone() for v in st['grams']]; count = st['metadata']['training_rows']
out = f'{SCRATCH}/exl3_k_{model}'; os.makedirs(out, exist_ok=True)
for K in Ks:
    values = []
    for i, weight in enumerate(w):
        qa = dict(K=K, devices=['cuda:0'], seed=91426, sigma_reg=.03, apply_out_scales=None, mul1=True)
        hdata = dict(H=grams[int(i == 2)].clone(), count=count, finalized=False, device=torch.device('cuda:0'))
        with torch.no_grad(): _, proxy, value = quantize_exl3(weight.T.contiguous(), hdata, qa, False, verbose=False)
        values.append(dict(shape=list(weight.shape), **{k: v.cpu() for k, v in value.items()}))
        get_temp_buffers.cache_clear(); del hdata, value; torch.cuda.empty_cache()
    write_legacy(f'{out}/expert_{K}.bin', values)
    from orbit_duet.exl3_adapter import EXL3Expert
    ref = EXL3Expert(f'{out}/expert_{K}.bin').decoded_weights()
    ev = eval_captures(model, w, {f'K{K}': (ref, [None] * 3)})
    print(K, os.path.getsize(f'{out}/expert_{K}.bin') * 8 / 37748736, {k: round(v[f'K{K}'], 3) for k, v in ev.items()}, flush=True)
