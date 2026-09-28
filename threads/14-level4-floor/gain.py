"""Candidate 2b: scaled mul1 residual (trellis gain gr) and base gain gb, lam 0.3, integer K."""
import json
from t14 import *
from orbit_duet.source import weights
L, E = 16, 36
Ws = weights('/tmp/nestquant/glm53-fp8-experts', L, E); Hs = hessians(L, E, Ws)
G = [(1.0, 0.85), (1.0, 0.92), (1.0, 1.0), (1.0, 1.08), (1.0, 1.15), (0.92, 1.0), (1.08, 1.0)]
meth = {}; prox = {}
for mi in range(3):
    P = problem(Ws[mi], Hs[mi], mi)
    for gb, gr in G:
        o = fit(P, lam=0.3, gb=gb, gr=gr)
        prox[f'{PROJ[mi]}_gb{gb}_gr{gr}'] = (o['l2'], o['l4']); print(PROJ[mi], gb, gr, o['l2'], o['l4'], flush=True)
        meth.setdefault(f'gb{gb}_gr{gr}@4', []).append(P.dequant(o['Q4']).bfloat16())
        if gr == 1.0: meth.setdefault(f'gb{gb}@2', []).append(P.dequant(o['Q2']).bfloat16())
    del P; torch.cuda.empty_cache()
meth = {k: [w.float() for w in v] for k, v in meth.items()}
res = evaluate(L, E, meth)
for k in sorted(res): print(f"{k:24s} routed {res[k]['routed']:7.3f} forced {res[k]['forced']:7.3f} ood {res[k]['ood']:7.3f}")
json.dump(dict(eval=res, proxy=prox), open('results_gain_l16_e36.json', 'w'), indent=1)
