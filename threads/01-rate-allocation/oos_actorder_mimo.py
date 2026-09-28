"""MiMo down: does the act-order(TP-shard) innovation spread survive out of sample? K-fold on the 32k training sample."""
import sys, json, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, '/home/coder/git/nestquant/threads/01-rate-allocation'); sys.path.insert(0, '/home/coder/git/orbit-duet')
from common import *
from orbit_duet.source import weights as load_weights
setup_gpu(); dev = 'cuda'
g, u, d = load_weights(MIMO_SRC, 55, 70, device=dev)
S = torch.load(stats_path('mimo', 55, 70) + '_training_sample.pt', weights_only=True, mmap=True)
X = S['x'].to(dev); P = S['p'].to(dev).float()
Z = (F.silu(F.linear(X, g.bfloat16())) * F.linear(X, u.bfloat16())).float()
Hx, Hh, outs, meta = load_grams('mimo', 55, 70); Hh = Hh.to(dev, torch.float64)
def gram(Z, w): Zw = (Z * w[:, None]).double(); return Zw.T @ Zw
def innov(Lt, H):
    Li = torch.linalg.solve_triangular(Lt, torch.eye(Lt.shape[0], device=dev, dtype=Lt.dtype), upper=False, unitriangular=True)
    M = Li @ H @ Li.T; m = H.shape[0] // 16
    return (torch.diagonal(M.view(m, 16, m, 16), dim1=0, dim2=2).permute(2, 0, 1).diagonal(dim1=1, dim2=2).sum(-1) / 16).cpu().numpy()
srt = torch.argsort(Hh.diagonal()).cpu(); perm = torch.cat([srt[j::8] for j in range(8)]).to(dev)
out = {}
for mode, pm in [('hadamard_only', None), ('ashard', perm)]:
    H = Hh if pm is None else Hh[pm][:, pm]; Zp = Z if pm is None else Z[:, pm]
    Hr, _ = rotate_H(H, seed=12); Lt, D = block_ldl(Hr); t_in = (D.diagonal(dim1=1, dim2=2).sum(-1) / 16).cpu().numpy()
    folds = torch.randperm(len(Zp), generator=torch.Generator().manual_seed(3)).to(dev).chunk(4); tcv = 0; tin_s = 0
    for k in range(4):
        tr = torch.cat([f for j, f in enumerate(folds) if j != k]); te = folds[k]
        Ha = gram(Zp[tr], P[tr]); Hb = gram(Zp[te], P[te])
        Lta, Da = block_ldl(rotate_H(Ha, seed=12)[0]); tcv = tcv + innov(Lta, rotate_H(Hb, seed=12, damp=0)[0]) / 4
        tin_s = tin_s + (Da.diagonal(dim1=1, dim2=2).sum(-1) / 16).cpu().numpy() / 4
    # per-shard AM/GM (what a TP-constrained allocator can exploit)
    def shard_amgm(t): return float(np.mean([amgm_db(t[j:j + 16]) for j in range(0, len(t), 16)]))
    out[mode] = dict(amgm_full_in=amgm_db(t_in), amgm_sample_in=amgm_db(tin_s), amgm_sample_cv=amgm_db(tcv),
                     shard_amgm_full_in=shard_amgm(t_in), shard_amgm_sample_cv=shard_amgm(tcv),
                     corr_full_in_vs_cv=float(np.corrcoef(np.log(t_in), np.log(tcv))[0, 1]))
    print(mode, out[mode], flush=True)
json.dump(out, open('results/oos_actorder_mimo_down.json', 'w'), indent=1)
