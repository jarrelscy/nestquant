"""9-expert confirmation: frozen base (blend 0.3, uniform K=2) + residual uniform 2/2/2 vs pattern 1.875/1.875/2.25."""
import sys, json
from t14 import *
from orbit_duet.source import weights
KR = {'uni': (2, 2, 2), 'pat': (1.875, 1.875, 2.25)}
for LE in sys.argv[1:]:
    L, E = map(int, LE.split(':'))
    outf = f'results_nine_l{L}_e{E}.json'
    if os.path.exists(outf): continue
    Ws = weights('/tmp/nestquant/glm53-fp8-experts', L, E); Hs = hessians(L, E, Ws)
    meth = {'EXL3-2': torch.load(f'{SCR}/exl3_l{L}_e{E}_K2.pt'), 'EXL3-4': torch.load(f'{SCR}/exl3_l{L}_e{E}_K4.pt')}
    prox = {}
    for mi in range(3):
        P = problem(Ws[mi], Hs[mi], mi)
        for tag, kr in KR.items():
            o = fit(P, lam=0.3, Kr=kr[mi])
            prox[f'{PROJ[mi]}/{tag}'] = (o['l2'], o['l4'])
            meth.setdefault(f'{tag}@4', []).append(P.dequant(o['Q4']).bfloat16().cpu())
            if tag == 'uni': meth.setdefault('base@2', []).append(P.dequant(o['Q2']).bfloat16().cpu())
            print(L, E, PROJ[mi], tag, o['l2'], o['l4'], flush=True)
        del P; torch.cuda.empty_cache()
    del Hs; torch.cuda.empty_cache()
    res = evaluate(L, E, {k: [w.cuda().float() for w in v] for k, v in meth.items()})
    for k, v in res.items(): print(L, E, k, v, flush=True)
    json.dump(dict(eval=res, proxy=prox), open(outf, 'w'), indent=1)
    del meth; torch.cuda.empty_cache()
