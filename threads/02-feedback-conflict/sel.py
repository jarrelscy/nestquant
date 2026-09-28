from common import *
from fbt import *
import sys, json
mi = int(sys.argv[1]); name=['gate','up','down'][mi]
Ws, Hs = glm(); P = Problem(Ws[mi], Hs[mi])
n2 = run_trellis(P, 'nat2'); n4 = run_trellis(P, 'nat4'); L2, L4 = n2['l2'], n4['l4']
out = {}
for mu in [0.3, 0.5, 0.7, 0.85]:
    r = fit_trellis_sel(P, mu, L2=L2, L4=L4)
    import collections
    print(name, f'blendsel mu={mu} r2 {r["l2"]/L2:.4f} r4 {r["l4"]/L4:.4f}', dict(collections.Counter(r['lams'])), flush=True)
    out[mu] = dict(l2=r['l2'], l4=r['l4'], L2=L2, L4=L4)
    sd = f'/tmp/nestquant/02-feedback-conflict/deq_trellis/{name}'; os.makedirs(sd, exist_ok=True)
    torch.save(dict(w2=P.dequant(r['Q2']).bfloat16().cpu(), w4=P.dequant(r['Q4']).bfloat16().cpu()), f'{sd}/blendsel_{mu}.pt')
json.dump(out, open(f'results/trellis_blendsel_glm_{name}.json','w'), indent=1)
