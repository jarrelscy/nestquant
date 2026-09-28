"""Frontier under thread-06 calibration settings. H built from training_sample rows with weight p^pw.
Configs: (pw2, sr0.03) = EXL3-matched, (pw1, sr0.3) = thread-06 GLM calibration. Splits: holdout (fit on 80%, thread-06 seed 606) and full."""
from common import *
from fbt import *
import json, torch.nn.functional as F
from orbit_duet.source import weights
Ws = weights('/tmp/nestquant/glm53-fp8-experts', 16, 36)
g, u, d = [w.cuda().float() for w in Ws]
ts = torch.load('/home/coder/git/orbit-duet/runs/glm53_pilot_matched_l16/statistics/l16_e36_training_sample.pt', map_location='cpu', mmap=True, weights_only=False)
n = len(ts['x']); perm = torch.randperm(n, generator=torch.Generator().manual_seed(606))
split = dict(holdout=perm[n // 5:].sort().values, full=torch.arange(n))
def grams_mix(idx, a=0.75):
    A = grams(idx, 2); U = grams(idx, 0)
    return [(1 - a) * A[k] / A[k].diagonal().mean() + a * U[k] / U[k].diagonal().mean() for k in range(3)]
def grams(idx, pw):
    Hx = torch.zeros(6144, 6144, device='cuda'); Ha = torch.zeros(2048, 2048, device='cuda')
    for i in range(0, len(idx), 2048):
        j = idx[i:i+2048]; x = ts['x'][j].cuda().float(); r = ts['p'][j].cuda().float().pow(pw / 2)[:, None]
        a = F.silu(x @ g.T) * (x @ u.T)
        Hx.addmm_((x*r).T, x*r); Ha.addmm_((a*r).T, a*r)
    return [Hx, Hx, Ha]
RULES = [('nat2','nat2',0,1.),('nat4','nat4',0,1.),('seq','seq',0,1.),('gam_0.0','blend',0,0.),('gam_0.5','blend',0,.5),
         ('blend_0.1','blend',.1,1.),('blend_0.2','blend',.2,1.),('blend_0.3','blend',.3,1.),('blend_0.5','blend',.5,1.),('blend_0.7','blend',.7,1.),('innov_exact_Minv','innov',0,1.)]
import sys
CFGS = [(2, 0.03), (1, 0.3)]; OUT = 'results/trellis_calib_frontier_glm.json'
if len(sys.argv) > 1 and sys.argv[1] == 'ext':
    RULES = [('nat2','nat2',0,1.),('nat4','nat4',0,1.),('gam_0.75','blend',0,.75),('gam_0.9','blend',0,.9),('b0.3_g0.75','blend',.3,.75),('b0.5_g0.75','blend',.5,.75),
             ('b0.2_g0.5','blend',.2,.5),('blend_0.3','blend',.3,1.),('blend_0.5','blend',.5,1.)]
    CFGS = [(1, 0.3), (1, 1.0)]; OUT = 'results/trellis_calib_frontier_ext_glm.json'
if len(sys.argv) > 1 and sys.argv[1] == 'mix':
    RULES = RULES + [('gam_0.75','blend',0,.75),('b0.3_g0.75','blend',.3,.75),('b0.5_g0.75','blend',.5,.75)]
    CFGS = [('mix', (0.5, 0.5, 1.0))]; OUT = f'results/trellis_calib_frontier_mix_glm_{sys.argv[2]}.json'
    split = {sys.argv[2]: split[sys.argv[2]]}
out = {}
for pw, sr in CFGS:
    for sp, idx in split.items():
        Hs = grams_mix(idx) if pw == 'mix' else grams(idx, pw); cfg = f'pw{pw}_sr{sr}_{sp}' if pw != 'mix' else f'mix_{sp}'
        for mi, name in enumerate(['gate','up','down']):
            torch.manual_seed(99 + mi)
            P = Problem(Ws[mi].cuda(), Hs[mi], damp=sr[mi] if isinstance(sr, tuple) else sr)
            sd = f'/tmp/nestquant/02-feedback-conflict/deq_{cfg}/{name}'; os.makedirs(sd, exist_ok=True)
            for tag, rule, lam, gam in RULES:
                r = run_trellis(P, rule, lam, gam=gam)
                out[f'{cfg}/{name}/{tag}'] = dict(l2=r['l2'], l4=r['l4'])
                print(f'{cfg} {name:5s} {tag:18s} l2 {r["l2"]:.6f} l4 {r["l4"]:.6f}', flush=True)
                q2 = r['Q4'] if rule == 'nat4' else r['Q2']
                torch.save(dict(w2=P.dequant(q2).bfloat16().cpu(), w4=P.dequant(r['Q4']).bfloat16().cpu()), f'{sd}/{tag}.pt')
            del P; torch.cuda.empty_cache()
            json.dump(out, open(OUT, 'w'), indent=1)
        if pw == 'mix' and sp == 'full':      # matched-calibration EXL3 anchors under the same H (thread-08 recipe)
            sys.path.insert(0, '/home/coder/git/nestquant/threads/05-exl3-harness'); import harness as hh
            for K in (2, 4):
                for mi, name in enumerate(['gate','up','down']):
                    Wq, info = hh.quantize_exl3_like(Ws[mi], Hs[mi], K, count=1, sigma_reg=sr[mi])
                    sd = f'/tmp/nestquant/02-feedback-conflict/deq_{cfg}/{name}'
                    torch.save(dict(w2=Wq.bfloat16().cpu(), w4=Wq.bfloat16().cpu()), f'{sd}/EXL3mix_{K}.pt')
                    print(cfg, name, 'EXL3', K, 'proxy', float(info['proxy']), flush=True)
                    torch.cuda.empty_cache()
        del Hs; torch.cuda.empty_cache()
