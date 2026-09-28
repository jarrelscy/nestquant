"""Step 2: block-LDL innovation statistics and ideal reverse-water-filling gains."""
import json, math, sys
import numpy as np, torch
sys.path.insert(0, '/home/coder/git/nestquant/threads/01-rate-allocation')
from common import *

setup_gpu()
dev = 'cuda'
EXPERTS = [('glm', L, E) for L in (16, 49, 66) for E in (36, 92, 165)] + [('mimo', 55, 70)]
out = {}
for model, L, E in EXPERTS:
    Hx, Hh, outs, meta = load_grams(model, L, E)
    name = f'{model}_l{L}_e{E}'
    out[name] = {}
    for proj, H in [('gate_up', Hx), ('down', Hh)]:
        H = H.to(dev, torch.float64)
        res = {}
        for variant in ['rot', 'norot']:
            if variant == 'rot':
                Hr, _ = rotate_H(H, seed=1)
            else:
                Hr = H.clone(); Hr.diagonal().add_(0.025 * Hr.diagonal().mean())
            Lt, D = block_ldl(Hr, 16)
            t = (D.diagonal(dim1=1, dim2=2).sum(-1) / 16).cpu().numpy()
            g = torch.exp(torch.linalg.slogdet(D)[1] / 16).cpu().numpy()
            Ls = torch.linalg.cholesky(Hr)
            d1 = (Ls.diagonal() ** 2).cpu().numpy()
            dg = Hr.diagonal().cpu().numpy()
            r = dict(amgm_db_block_trace=amgm_db(t), amgm_db_block_det=amgm_db(g), amgm_db_scalar_ldl=amgm_db(d1),
                     amgm_db_diag=amgm_db(dg),
                     # fraction of total loss weight held by top 10% blocks
                     top10pct_trace_share=float(np.sort(t)[::-1][:max(1, len(t) // 10)].sum() / t.sum()),
                     t_over_first=[float(x) for x in (t / t.mean())[[0, len(t) // 4, len(t) // 2, 3 * len(t) // 4, len(t) - 1]]])
            wf = {}
            for Rb in (2, 3, 4):
                R = waterfill(t, Rb, 0, 8)
                wf[str(Rb)] = dict(
                    cont_0_8_db=10 * math.log10(dist_model(t, np.full(len(t), Rb)) / dist_model(t, R)),
                    int_1_8_db=10 * math.log10(dist_model(t, np.full(len(t), Rb)) / dist_model(t, greedy_int(t, Rb, 1, 8, 1.0))),
                    half_1_8_db=10 * math.log10(dist_model(t, np.full(len(t), Rb)) / dist_model(t, greedy_int(t, Rb, 1, 8, 0.5))),
                    Rmin=float(R.min()), Rmax=float(R.max()))
            r['waterfill_block_trace'] = wf
            res[variant] = r
            if variant == 'rot':
                np.save(f'/tmp/nestquant/01-rate-allocation/t_{name}_{proj}.npy', t)
        out[name][proj] = res
        rr = res['rot']
        print(name, proj, 'rot: AM/GM trace %.2f det %.2f scalar %.2f diag %.2f | WF2 %.2f WF3 %.2f WF4 %.2f int2 %.2f int4 %.2f | norot trace %.2f' % (
            rr['amgm_db_block_trace'], rr['amgm_db_block_det'], rr['amgm_db_scalar_ldl'], rr['amgm_db_diag'],
            rr['waterfill_block_trace']['2']['cont_0_8_db'], rr['waterfill_block_trace']['3']['cont_0_8_db'], rr['waterfill_block_trace']['4']['cont_0_8_db'],
            rr['waterfill_block_trace']['2']['int_1_8_db'], rr['waterfill_block_trace']['4']['int_1_8_db'],
            res['norot']['amgm_db_block_trace']), flush=True)
        del H
    torch.cuda.empty_cache()
json.dump(out, open('/home/coder/git/nestquant/threads/01-rate-allocation/ldl_stats.json', 'w'), indent=1)
