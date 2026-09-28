"""Step 5: nestedness + metadata of greedy per-shard profiles for MiMo down (act-order-shard) and GLM (CV innovations)."""
import sys, json, math, heapq, pickle, numpy as np, torch
sys.path.insert(0, '/home/coder/git/nestquant/threads/01-rate-allocation')
from common import *
CB, DN = pickle.load(open('/tmp/nestquant/01-rate-allocation/lloyd_max.pkl', 'rb'))
def greedy(a, Rbar, allowed):
    a = np.asarray(a, np.float64); m = len(a); pos = np.zeros(m, int); budget = Rbar * m - m * math.log2(allowed[0]); spent = 0
    def gain(i):
        if pos[i] + 1 >= len(allowed): return None
        n0, n1 = allowed[pos[i]], allowed[pos[i] + 1]; return a[i] * (DN[n0] - DN[n1]) / math.log2(n1 / n0), math.log2(n1 / n0)
    h = [(-gain(i)[0], i, gain(i)[1]) for i in range(m)]; heapq.heapify(h)
    while h:
        _, i, c = heapq.heappop(h)
        if spent + c > budget + 1e-9: continue
        spent += c; pos[i] += 1; g = gain(i)
        if g: heapq.heappush(h, (-g[0], i, g[1]))
    return np.log2([allowed[k] for k in pos])
INT = [2 ** r for r in range(9)]
# MiMo down, act-order-shard in-sample t (full 408k gram)
Hx, Hh, outs, meta = load_grams('mimo', 55, 70); Hh = Hh.double()
srt = torch.argsort(Hh.diagonal()); perm = torch.cat([srt[j::8] for j in range(8)])
Lt, D = block_ldl(rotate_H(Hh[perm][:, perm], seed=12)[0]); t = (D.diagonal(dim1=1, dim2=2).sum(-1) / 16).numpy()
out = {}
for lad_name, lad in [('int', INT)]:
    prof = {R: np.concatenate([greedy(t[j:j + 16], R, lad) for j in range(0, 128, 16)]) for R in (2, 3, 4)}
    nested = bool(np.all(prof[3] >= prof[2]) and np.all(prof[4] >= prof[3]))
    inc = {f'{a}->{b}': np.unique(prof[b] - prof[a], return_counts=True) for a, b in [(2, 3), (3, 4)]}
    out[lad_name] = dict(nested=nested, hist={R: dict(zip(*[x.tolist() for x in np.unique(p, return_counts=True)])) for R, p in prof.items()},
                         increments={k: dict(zip(*[x.tolist() for x in v])) for k, v in inc.items()},
                         shard_means={R: [float(p[j:j + 16].mean()) for j in range(0, 128, 16)] for R, p in prof.items()})
    print(lad_name, json.dumps(out[lad_name]))
# metadata: per block, per level, 4-bit rate code; down has 128 blocks, weights 6144*2048
meta_bits = 128 * 3 * 4
out['metadata_bpw_down'] = meta_bits / (6144 * 2048)
print('metadata bpw (down, 3 levels x 4-bit code per 16-col block):', out['metadata_bpw_down'])
json.dump(out, open('results/nested_mimo_down.json', 'w'), indent=1, default=str)
