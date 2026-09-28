from common import *
from fbt import *
import json, sys, time
from common import glm
Ws, Hs = glm()
out = {}
for damp in [0.025, 0.3, 1.0]:
    for mi, name in enumerate(['gate','up','down']):
        torch.manual_seed(99 + mi)
        P = Problem(Ws[mi], Hs[mi], damp=damp)
        sd = f'/tmp/nestquant/02-feedback-conflict/deq_trellis_d{damp}/{name}'; os.makedirs(sd, exist_ok=True)
        for tag, rule, lam in [('nat2','nat2',0),('nat4','nat4',0),('blend_0.3','blend',.3),('blend_0.5','blend',.5),('innov_exact_Minv','innov',0)]:
            r = run_trellis(P, rule, lam)
            out[f'{damp}/{name}/{tag}'] = dict(l2=r['l2'], l4=r['l4'])
            print(f'd{damp} {name:5s} {tag:18s} l2 {r["l2"]:.6f} l4 {r["l4"]:.6f}', flush=True)
            w2 = P.dequant(torch.nan_to_num(r['Q2'])) if rule != 'nat4' else P.dequant(r['Q4'])
            torch.save(dict(w2=w2.bfloat16().cpu(), w4=P.dequant(r['Q4']).bfloat16().cpu()), f'{sd}/{tag}.pt')
        del P; torch.cuda.empty_cache()
        json.dump(out, open('results/trellis_damp_sweep_glm.json', 'w'), indent=1)
