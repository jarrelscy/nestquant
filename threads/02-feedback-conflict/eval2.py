"""Score calib-frontier: held-out 20% training rows (p^2-weighted, thread-06 split) for holdout fits; matched capture (harness evaluate) for full fits.
Anchors: thread-06 EXL3 fits at the same calibration, NVFP4."""
import sys, json, torch
sys.path.insert(0, '/home/coder/git/nestquant/threads/05-exl3-harness'); sys.path.insert(0, '/home/coder/git/orbit-duet')
import harness as h; h.gpu_cap(12)
from orbit_duet.evaluate import teacher
from pathlib import Path
data = h.load_expert(16, 36); native = data.teacher
ts = torch.load('/home/coder/git/orbit-duet/runs/glm53_pilot_matched_l16/statistics/l16_e36_training_sample.pt', map_location='cpu', mmap=True, weights_only=False)
n = len(ts['x']); perm = torch.randperm(n, generator=torch.Generator().manual_seed(606)); val = perm[:n // 5].sort().values
TAGS = ['nat2','nat4','seq','gam_0.0','gam_0.5','blend_0.1','blend_0.2','blend_0.3','blend_0.5','blend_0.7','innov_exact_Minv']
T6 = '/tmp/nestquant/06-expert-objective'
def meth(cfg, sp):
    m = {}
    for t in TAGS:
        parts = [torch.load(f'/tmp/nestquant/02-feedback-conflict/deq_{cfg}_{sp}/{x}/{t}.pt') for x in ['gate','up','down']]
        m[f'{t}@2'] = [p['w2'].cuda() for p in parts]; m[f'{t}@4'] = [p['w4'].cuda() for p in parts]
    for b in [2, 4]:
        if cfg == 'mix' and sp == 'full':
            m[f'EXL3-{b}'] = [torch.load(f'/tmp/nestquant/02-feedback-conflict/deq_mix_full/{x}/EXL3mix_{b}.pt')['w4'].cuda() for x in ['gate','up','down']]
        if cfg not in ('pw1_sr1.0', 'mix'): m[f'EXL3-{b}'] = [torch.load(f'{T6}/glm_{sp}/{x}_b{b}_{cfg}_auto_H.pt', map_location='cuda') for x in 'gud']
    return m
def holdout(methods):
    acc = {k: [0., 0.] for k in methods}
    with torch.no_grad():
        for a in range(0, len(val), 1024):
            j = val[a:a+1024]; x = ts['x'][j].cuda(); p2 = ts['p'][j].cuda().double().square()
            y = teacher(x, native).double(); den = float((y.square().sum(-1) * p2).sum())
            for k, w in methods.items():
                e = (teacher(x, w).double() - y).square().sum(-1); acc[k][0] += float((e * p2).sum()); acc[k][1] += den
    return {k: 100 * (v[0] / v[1]) ** .5 for k, v in acc.items()}
CFGS = ['pw2_sr0.03', 'pw1_sr0.3']; OUT = 'results/output_calib_frontier.json'
if len(sys.argv) > 1 and sys.argv[1] == 'ext':
    TAGS = ['nat2','nat4','gam_0.75','gam_0.9','b0.3_g0.75','b0.5_g0.75','b0.2_g0.5','blend_0.3','blend_0.5']; CFGS = ['pw1_sr0.3', 'pw1_sr1.0']; OUT = 'results/output_calib_frontier_ext.json'
if len(sys.argv) > 1 and sys.argv[1] == 'mix':
    TAGS = TAGS + ['gam_0.75','b0.3_g0.75','b0.5_g0.75']; CFGS = ['mix']; OUT = 'results/output_calib_frontier_mix.json'
out = {}
for cfg in CFGS:
    ho = holdout(meth(cfg, 'holdout')); torch.cuda.empty_cache()
    mf = meth(cfg, 'full')
    if cfg in ('pw2_sr0.03', 'mix'): mf['nvfp4'] = h.load_nvfp4('/home/coder/git/orbit-duet/runs/glm53_pilot_matched_l16/nvfp4_e36/weights.pt', data)
    tab = h.table(h.evaluate(data, mf)); del mf; torch.cuda.empty_cache()
    out[cfg] = dict(holdout=ho, capture=tab)
    for k in tab:
        t = tab[k]; print(f'{cfg} {k:22s} hold {ho.get(k, float("nan")):7.3f} routed {t["all/routed"]:7.3f} forced {t["all/forced"]:7.3f} ood {t["ood/forced"]:7.3f}', flush=True)
    json.dump(out, open(OUT, 'w'), indent=1)
