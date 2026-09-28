from common import *
from fbt import *
import json, sys, time
model, mi = sys.argv[1], int(sys.argv[2]); name = ['gate','up','down'][mi]
torch.manual_seed(99 + mi)
if model == 'glm':
    from common import glm; Ws, Hs = glm(); W, H = Ws[mi], Hs[mi]
else:
    Hs = mimo_H(); H = Hs[mi]; W = torch.randn((2048, 6144) if mi < 2 else (6144, 2048), device='cuda')
P = Problem(W, H)
sd = f'/tmp/nestquant/02-feedback-conflict/deq_trellis/{name}'; os.makedirs(sd, exist_ok=True)
res = dict(model=model, matrix=name, rows={}); t0 = time.time()
for tag, rule, lam in [('nat2','nat2',0),('nat4','nat4',0),('seq','seq',0),('blend_0.3','blend',.3),('blend_0.5','blend',.5),
                       ('blend_0.7','blend',.7),('blend_0.9','blend',.9),('innov_exact_Minv','innov',0)]:
    r = run_trellis(P, rule, lam)
    res['rows'][tag] = dict(l2=r['l2'], l4=r['l4'])
    print(f'{tag:20s} l2 {r["l2"]:.6f} l4 {r["l4"]:.6f}', flush=True)
    if model == 'glm':
        torch.save(dict(w2=P.dequant(torch.nan_to_num(r['Q2'])).bfloat16().cpu(), w4=P.dequant(r['Q4']).bfloat16().cpu()), f'{sd}/{tag}.pt')
res['seconds'] = time.time() - t0
json.dump(res, open(f'/home/coder/git/nestquant/threads/02-feedback-conflict/results/trellis_{model}_{name}.json', 'w'), indent=1)
