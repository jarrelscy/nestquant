"""Out-of-sample block innovations: in-sample D vs K-fold cross-validated D vs eval-capture D (diagnostic only)."""
import json, math, sys
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, '/home/coder/git/nestquant/threads/01-rate-allocation'); sys.path.insert(0, '/home/coder/git/orbit-duet')
from common import *
from orbit_duet.source import weights as load_weights
setup_gpu(); dev = 'cuda'
model, L, E = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); K = 4
damp = 0.025
name = f'{model}_l{L}_e{E}'
S = torch.load(stats_path(model, L, E) + '_training_sample.pt', weights_only=True, mmap=True)
Hx, Hh, outs, meta = load_grams(model, L, E)
X = S['x'].to(dev).float(); P = S['p'].to(dev).float()
src = GLM_SRC if model == 'glm' else MIMO_SRC
res = dict(expert=name, rows=len(X), gram_rows=meta['training_rows'])
wts = None
try:
    g, u, d = load_weights(src, L, E, device=dev)
    wts = (g, u, d)
except Exception as ex:
    print('no weights:', ex)

def feats(Xb, proj):
    if proj == 'gate_up': return Xb
    g, u, d = wts
    return (F.silu(F.linear(Xb.bfloat16(), g.bfloat16())) * F.linear(Xb.bfloat16(), u.bfloat16())).float()

def gram(Z, w):
    Zw = (Z * w[:, None]).double(); return Zw.T @ Zw

def innov(Lt, H):
    """block diag of Lt^{-1} H Lt^{-T} traces / 16"""
    Li = torch.linalg.solve_triangular(Lt, torch.eye(Lt.shape[0], device=dev, dtype=Lt.dtype), upper=False, unitriangular=True)
    M = Li @ H @ Li.T
    m = H.shape[0] // 16
    Db = torch.diagonal(M.view(m, 16, m, 16), dim1=0, dim2=2).permute(2, 0, 1)
    return (Db.diagonal(dim1=1, dim2=2).sum(-1) / 16).cpu().numpy()

caps = ['glm53_matched_context_pilot_v1_capture', 'native_id_control_v1_capture', 'ood_controlled_v1_capture']
for proj in (['gate_up', 'down'] if wts is not None else ['gate_up']):
    Hfull = Hx.to(dev, torch.float64) if proj == 'gate_up' else Hh.to(dev, torch.float64)
    Z = feats(X, proj)
    Hs = gram(Z, P)
    rel = float((Hs - Hfull).norm() / Hfull.norm())
    seed = 11 if proj == 'gate_up' else 12
    Hr, su = rotate_H(Hfull, seed=seed, damp=damp)
    Lt, D = block_ldl(Hr, 16)
    t_in = (D.diagonal(dim1=1, dim2=2).sum(-1) / 16).cpu().numpy()
    # K-fold CV: fit Lt on K-1 folds (damped), measure innovation on held-out fold (undamped), same rotation
    perm = torch.randperm(len(Z), generator=torch.Generator().manual_seed(3)).to(dev)
    folds = perm.chunk(K)
    t_cv = np.zeros_like(t_in); t_cvin = np.zeros_like(t_in)
    for k in range(K):
        tr = torch.cat([f for j, f in enumerate(folds) if j != k]); te = folds[k]
        Ha = gram(Z[tr], P[tr]); Hb = gram(Z[te], P[te])
        Har, _ = rotate_H(Ha, seed=seed, damp=damp)
        Hbr, _ = rotate_H(Hb, seed=seed, damp=0.0)
        Lta, Da = block_ldl(Har, 16)
        sc = float(Ha.diagonal().mean() / Hb.diagonal().mean())
        t_cv += innov(Lta, Hbr) * sc / K
        t_cvin += (Da.diagonal(dim1=1, dim2=2).sum(-1) / 16).cpu().numpy() / K
    np.save(f'/tmp/nestquant/01-rate-allocation/tcv_{name}_{proj}.npy', t_cv)
    r = dict(sample_vs_gram_relerr=rel, amgm_in=amgm_db(t_in), amgm_cv=amgm_db(t_cv), amgm_cv_insample=amgm_db(t_cvin),
             corr_in_cv=float(np.corrcoef(np.log(t_in), np.log(t_cv))[0, 1]),
             chunks_in=[round(float(c), 3) for c in np.array_split(t_in / t_in.mean(), 8)[0:8] for c in [c.mean()]],
             chunks_cv=[round(float(c), 3) for c in np.array_split(t_cv / t_cv.mean(), 8) for c in [c.mean()]],
             oos_over_in_ratio_mean=float((t_cv / t_cvin).mean()))
    Hsr, _ = rotate_H(Hs, seed=seed, damp=0.0)
    ts = innov(Lt, Hsr) * float(Hfull.diagonal().mean() / Hs.diagonal().mean())
    r['amgm_fullL_on_sample'] = amgm_db(ts); r['corr_in_fullL_sample'] = float(np.corrcoef(np.log(t_in), np.log(ts))[0, 1])
    np.save(f'/tmp/nestquant/01-rate-allocation/tsamp_{name}_{proj}.npy', ts)
    if model == 'glm':
        Hrt, _ = rotate_H(Hfull, seed=seed, damp=0.0)
        for c in caps:
            import os
            if not os.path.exists(f'{ROOT}/runs/{c}/layer_{L}.pt'): continue
            C = torch.load(f'{ROOT}/runs/{c}/layer_{L}.pt', weights_only=True, mmap=True)
            Zc = feats(C['x'].to(dev).float(), proj)
            He = gram(Zc, torch.ones(len(Zc), device=dev))
            Her, _ = rotate_H(He, seed=seed, damp=0.0)
            sc = float(Hfull.diagonal().mean() / He.diagonal().mean())
            te = innov(Lt, Her) * sc
            r[f'amgm_eval_{c}'] = amgm_db(te)
            r[f'chunks_eval_{c}'] = [round(float(c_.mean()), 3) for c_ in np.array_split(te / te.mean(), 8)]
            r[f'corr_cv_eval_{c}'] = float(np.corrcoef(np.log(t_cv), np.log(te))[0, 1])
            r[f'corr_in_eval_{c}'] = float(np.corrcoef(np.log(t_in), np.log(te))[0, 1])
    res[proj] = r
    print(name, proj, json.dumps(r), flush=True)
json.dump(res, open(f'/home/coder/git/nestquant/threads/01-rate-allocation/results/oos_{name}.json', 'w'), indent=1)
