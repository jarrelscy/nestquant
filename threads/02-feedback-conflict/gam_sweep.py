"""Base feedback gain gamma (base target w - gam*E2 M; gam=0 = base without LDLQ feedback, thread-03 point) x blend lam, at damping 0.025 / 0.3."""
from common import *
from fbt import *
import json
from common import glm
Ws, Hs = glm()
out = {}
for damp in [0.025, 0.3]:
    for mi, name in enumerate(['gate','up','down']):
        torch.manual_seed(99 + mi)
        P = Problem(Ws[mi], Hs[mi], damp=damp)
        sd = f'/tmp/nestquant/02-feedback-conflict/deq_trellis_d{damp}/{name}'; os.makedirs(sd, exist_ok=True)
        for tag, lam, gam in [('seq',0,1.),('gam_0.0',0,0.),('gam_0.25',0,.25),('gam_0.5',0,.5),('gam_0.75',0,.75),('blend_0.1',.1,1.),('blend_0.2',.2,1.)]:
            r = run_trellis(P, 'blend', lam, gam=gam)
            out[f'{damp}/{name}/{tag}'] = dict(l2=r['l2'], l4=r['l4'])
            print(f'd{damp} {name:5s} {tag:18s} l2 {r["l2"]:.6f} l4 {r["l4"]:.6f}', flush=True)
            torch.save(dict(w2=P.dequant(r['Q2']).bfloat16().cpu(), w4=P.dequant(r['Q4']).bfloat16().cpu()), f'{sd}/{tag}.pt')
        del P; torch.cuda.empty_cache()
        json.dump(out, open('results/trellis_gam_sweep_glm.json', 'w'), indent=1)
