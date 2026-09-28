"""Per-projection lambda: fit lam grid (b2/r2), measure per-projection output-MSE contributions (others
teacher), pick combos under L2 <= 34.59 assuming additivity, then verify combos jointly."""
import json, itertools
from t14 import *
from orbit_duet.source import weights
L, E = 16, 36
Ws = weights('/tmp/nestquant/glm53-fp8-experts', L, E); Hs = hessians(L, E, Ws)
LAMS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
def get(mi, lam):
    f = f'{SCR}/fit_l{L}_e{E}_{PROJ[mi]}_lam{lam}_kb2_kr2.pt'
    if os.path.exists(f): return torch.load(f)
    P = problem(Ws[mi], Hs[mi], mi); o = fit(P, lam=lam)
    r = dict(w2=P.dequant(o['Q2']).bfloat16().cpu(), w4=P.dequant(o['Q4']).bfloat16().cpu(), l2=o['l2'], l4=o['l4'])
    torch.save(r, f); del P; torch.cuda.empty_cache(); return r
fits = {(mi, lam): get(mi, lam) for mi in range(3) for lam in LAMS}
T = [w.cpu().bfloat16() for w in data(L, E).teacher]
m = {}
for (mi, lam), r in fits.items():
    for lv in ('w2', 'w4'):
        m[f'{mi}|{lam}|{lv}'] = [r[lv] if j == mi else T[j] for j in range(3)]
res = {}; names = list(m)
for a in range(0, len(names), 8):
    res.update(evaluate(L, E, {k: [w.cuda().float() for w in m[k]] for k in names[a:a+8]})); torch.cuda.empty_cache()
json.dump(res, open('results_lamp_contrib.json', 'w'), indent=1)
C = {k: {q: v[q]**2 for q in ('routed', 'forced', 'ood')} for k, v in res.items()}
cands = []
for combo in itertools.product(LAMS, repeat=3):
    l2 = sum(C[f'{mi}|{combo[mi]}|w2']['routed'] for mi in range(3)) ** .5
    l4 = sum(C[f'{mi}|{combo[mi]}|w4']['routed'] for mi in range(3)) ** .5
    cands.append((l4, l2, combo))
ok = sorted([c for c in cands if c[1] <= 34.59 * 1.0])[:6]
for c in ok: print('pred', c)
ver = {f'lam{c[2]}@{lv}': [fits[(mi, c[2][mi])][f'w{lv}'] for mi in range(3)] for c in ok for lv in (2, 4)}
ver['lam0.3x3@4'] = [fits[(mi, 0.3)]['w4'] for mi in range(3)]
vr = evaluate(L, E, {k: [w.cuda().float() for w in v] for k, v in ver.items()})
for k, v in vr.items(): print(k, v)
json.dump(dict(pred=ok, verify=vr), open('results_lamp.json', 'w'), indent=1)
