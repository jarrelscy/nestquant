from t14 import *
import json, time
from orbit_duet.source import weights
L, E = 16, 36
Ws = weights('/tmp/nestquant/glm53-fp8-experts', L, E)
Hs = hessians(L, E, Ws)
meth = {}; diag = {}
for mi in range(3):
    t0 = time.time()
    P = problem(Ws[mi], Hs[mi], mi)
    o = fit(P, lam=0.3, collect=(True))
    n4 = fit(P, nat=4, gb=0.9); n2 = fit(P, nat=2)
    print(PROJ[mi], 'blend0.3', o['l2'], o['l4'], 'nat4', n4['l4'], 'nat2', n2['l4'], f'{time.time()-t0:.0f}s', flush=True)
    for k, q in [('b03@2', o['Q2']), ('b03@4', o['Q4']), ('nat4', n4['Q4']), ('nat2', n2['Q4'])]:
        meth.setdefault(k, []).append(P.dequant(q))
    torch.save(o['col'], f'{SCR}/col_{PROJ[mi]}.pt')
    del P, o; torch.cuda.empty_cache()
res = evaluate(L, E, meth)
for k, v in res.items(): print(k, v)
