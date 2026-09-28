"""Full frontier for one matrix. usage: frontier.py {glm,glmgauss,mimogauss} {0,1,2}"""
from common import *
from fb import *
import json, sys, time
model, mi = sys.argv[1], int(sys.argv[2])
name = ['gate','up','down'][mi]
torch.manual_seed(1234 + mi)
if model == 'glm':
    Ws, Hs = glm(); W, H = Ws[mi], Hs[mi]
elif model == 'glmgauss':
    Ws, Hs = glm(); H = Hs[mi]; W = torch.randn(Ws[mi].shape, device='cuda')
else:
    Hs = mimo_H(); H = Hs[mi]; W = torch.randn((2048, 6144) if mi < 2 else (6144, 2048), device='cuda')
save = model == 'glm'
sd = f'/tmp/nestquant/02-feedback-conflict/deq/{name}'; os.makedirs(sd, exist_ok=True)
P = Problem(W, H)
res = dict(model=model, matrix=name, AMGM_D=float(P.D.mean()/P.D.log().mean().exp()), rows={})
def keep(tag, l2, l4, Q2=None, Q4=None, extra=None):
    res['rows'][tag] = dict(l2=l2, l4=l4, **(extra or {}))
    print(f'{tag:28s} l2 {l2:.6f} l4 {l4:.6f}', flush=True)
    if save and Q2 is not None:
        torch.save(dict(w2=P.dequant(Q2).bfloat16().cpu(), w4=P.dequant(Q4).bfloat16().cpu()), f'{sd}/{tag}.pt')
t0 = time.time()
n2 = run(P, 'nat2'); n4 = run(P, 'nat4'); L2, L4 = n2['l2'], n4['l4']
keep('nat2', n2['l2'], n2['l4'], n2['Q2'], n2['Q4']); keep('nat4', n4['l2'], n4['l4'], n4['Q2'], n4['Q4'])
res['rows']['seq'] = res['rows']['nat2'].copy()
Q2, _, _ = cd_joint(P, n2['idx'], n2['c2'], n2['c4'], 1.0, 0.0); l2cd = P.loss(Q2)
_, Q4, _ = cd_joint(P, n4['idx'], n4['c2'], n4['c4'], 0.0, 1.0); l4cd = P.loss(Q4)
keep('nat2_cd', l2cd, float('nan')); keep('nat4_cd', float('nan'), l4cd)
for lam in [0.3, 0.5, 0.7, 0.9]:
    r = run(P, 'blend', lam); keep(f'blend_{lam}', r['l2'], r['l4'], r['Q2'] if lam == 0.5 else None, r['Q4'])
for mu in [0.5, 0.9]:
    r = run(P, 'shared', mu, mu=mu); keep(f'matgptq_{mu}', r['l2'], r['l4'])
for mu in [0.3, 0.5, 0.7, 0.85, 0.95]:
    r = run(P, 'joint', mu); keep(f'joint_{mu}', r['l2'], r['l4'])
    if mu >= 0.5:
        Q2, Q4, _ = cd_joint(P, r['idx'], r['c2'], r['c4'], (1-mu)/L2, mu/L4)
        keep(f'jointcd_{mu}', P.loss(Q2), P.loss(Q4), Q2, Q4)
r = innov_refine(P, torch.tensor(n2['c2']), torch.tensor(n2['c4']))
keep('innov_exact_Minv', r['l2'], r['l4'], r['Q2'], r['Q4'])
res['native'] = dict(l2=L2, l4=L4, l2_cd=l2cd, l4_cd=l4cd)
res['seconds'] = time.time() - t0; res['peak_cuda_gib'] = torch.cuda.max_memory_allocated() / 2**30
json.dump(res, open(f'/home/coder/git/nestquant/threads/02-feedback-conflict/results/frontier_{model}_{name}.json', 'w'), indent=1)
print('done', res['seconds'], res['peak_cuda_gib'])
