"""MiMo: (a) can layer L's router input predict layer L+1's experts (prefetch)? (b) LRU with concurrent streams."""
import json, collections, numpy as np, torch
from safetensors import safe_open
torch.set_num_threads(8)
SRC = '/tmp/mimo-a100/data/jarrel/mimo-exl3-sequential/source'
MI = '/tmp/mimo-a100/data/jarrel/mimo-exl3-repair23-sequential'
idx = json.load(open(f'{SRC}/model.safetensors.index.json'))['weight_map']
def get(k):
    with safe_open(f'{SRC}/{idx[k]}', 'pt') as f: return f.get_tensor(k)
G = {L: (get(f'model.layers.{L}.mlp.gate.weight').float(), get(f'model.layers.{L}.mlp.gate.e_score_correction_bias').float(),
         get(f'model.layers.{L}.post_attention_layernorm.weight').float()) for L in [65, 66, 67]}
def route(x, L, k=8):
    W, b, _ = G[L]; s = torch.sigmoid(x.float() @ W.T)
    return torch.topk(s + b, k, dim=-1).indices
res = {}
for rk in [0, 1]:
    D = {L: torch.load(f'{MI}/capture{L}/rank{rk}.pt', map_location='cpu', mmap=True, weights_only=False) for L in [65, 66, 67]}
    assert all(torch.equal(D[65]['sequence_ids'], D[L]['sequence_ids']) for L in [66, 67])
    n = 4096
    for L in [65, 66, 67]:
        x = D[L]['x'][:n]; ids = D[L]['topk_ids'][:n]
        pr = route(x, L)
        same = np.mean([len(set(a.tolist()) & set(b.tolist())) / 8 for a, b in zip(pr, ids)])
        res.setdefault(f'self_reproduce_L{L}', []).append(float(same))
    for L in [65, 66]:
        x = D[L]['x'][:n]; tgt = D[L+1]['topk_ids'][:n]
        # x is normed with norm_L; re-weight to norm_{L+1}
        xr = x.float() / G[L][2] * G[L+1][2]
        for k in [8, 12, 16, 24, 32]:
            for nm, xx in [('raw', x), ('renorm', xr)]:
                pr = route(xx, L+1, k)
                rec = np.mean([len(set(a.tolist()) & set(b.tolist())) / 8 for a, b in zip(pr, tgt)])
                res.setdefault(f'predict_L{L+1}_from_L{L}_{nm}_top{k}_recall', []).append(float(rec))
        # two layers ahead
        if L == 65:
            for k in [8, 16, 32]:
                pr = route(x, 67, k); tgt2 = D[67]['topk_ids'][:n]
                rec = np.mean([len(set(a.tolist()) & set(b.tolist())) / 8 for a, b in zip(pr, tgt2)])
                res.setdefault(f'predict_L67_from_L65_top{k}_recall', []).append(float(rec))
    del D
res = {k: float(np.mean(v)) for k, v in res.items()}
# LRU with N interleaved streams, MiMo L66, full 8 ranks
I = []; S = []
for rk in range(8):
    d = torch.load(f'{MI}/capture66/rank{rk}.pt', map_location='cpu', mmap=True, weights_only=False)
    I.append(d['topk_ids'].numpy().copy()); S.append(d['sequence_ids'].numpy().copy() + rk * 10**7); del d
ids = np.concatenate(I); seq = np.concatenate(S)
seqs = [ids[seq == s] for s in np.unique(seq)]
L = min(len(s) for s in seqs)
def lru(streams, C):
    cache = collections.OrderedDict(); hits = tot = 0; misses_per_step = []
    for t in range(L):
        m = 0; batch = set()
        for s in streams: batch.update(s[t].tolist())
        for e in batch:
            tot += 1
            if e in cache: hits += 1; cache.move_to_end(e)
            else:
                m += 1; cache[e] = 1
                if len(cache) > C: cache.popitem(last=False)
        misses_per_step.append(m)
    return hits / tot, float(np.mean(misses_per_step)), float(tot / L)
out = {}
for N in [1, 4, 16, 64]:
    for C in [48, 96, 192]:
        hs = []; ms = []; us = []
        for g in range(0, min(len(seqs), 64) - N + 1, max(N, 8)):
            h, m, u = lru(seqs[g:g+N], C); hs.append(h); ms.append(m); us.append(u)
        out[f'N{N}_C{C}'] = dict(hit=float(np.mean(hs)), misses_per_step=float(np.mean(ms)), unique_experts_per_step=float(np.mean(us)))
res['lru_concurrent_L66'] = out
print(json.dumps(res, indent=1))
json.dump(res, open('/home/coder/git/nestquant/threads/10-streaming-system/predict_next.json', 'w'), indent=1)
