"""Expert popularity skew, split-half/OOD stability, temporal locality and LRU cache hit rates from routing captures."""
import json, sys, collections, numpy as np, torch
torch.set_num_threads(8)
OD = '/home/coder/git/orbit-duet/runs'
MI = '/tmp/mimo-a100/data/jarrel/mimo-exl3-repair23-sequential'

def load_glm(path):
    d = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    return d['ids'].numpy().copy(), d['p'].numpy().copy(), d['document_ids'].numpy().copy(), d['token_positions'].numpy().copy(), d.get('domains')

def coverage(counts, fracs):
    s = np.sort(counts)[::-1]; c = np.cumsum(s) / s.sum(); E = len(s)
    return {f: float(c[max(1, int(round(f*E)))-1]) for f in fracs}

def lru_sim(ids, docs, E, sizes, static=None):
    """Per-layer 'upgraded' cache of C experts; sequential decode order within docs.
    Returns hit rate (fraction of routes whose expert is already upgraded)."""
    out = {}
    for C in sizes:
        hits = tot = 0
        for d in np.unique(docs):
            rows = ids[docs == d]
            cache = collections.OrderedDict()
            if static is not None:
                for e in static[:C]: cache[int(e)] = 1
            for r in rows:
                for e in r:
                    e = int(e); tot += 1
                    if e in cache: hits += 1; cache.move_to_end(e)
                    else:
                        cache[e] = 1
                        if len(cache) > C: cache.popitem(last=False)
        out[C] = hits / tot
    return out

def static_hit(ids, top):
    return float(np.isin(ids, top).mean())

def analyse(ids, p, docs, E, label, sizes, extra_split=None):
    counts = np.bincount(ids.ravel(), minlength=E).astype(float)
    mass = np.bincount(ids.ravel(), weights=p.ravel(), minlength=E)
    T = len(ids); mean = counts.mean()
    pr = counts / counts.sum()
    ent = -(pr[pr > 0] * np.log(pr[pr > 0])).sum()
    r = dict(label=label, tokens=int(T), experts=E, max_over_mean=float(counts.max()/mean), zero_experts=int((counts == 0).sum()),
             min_over_mean=float(counts.min()/mean), eff_experts_frac=float(np.exp(ent)/E),
             hit_coverage=coverage(counts, [0.05, 0.1, 0.25, 0.37, 0.5]), mass_coverage=coverage(mass, [0.1, 0.25, 0.5]))
    # poisson noise reference: expected max/mean for uniform routing at this sample size
    lam = T * 8 / E; r['uniform_expected_cv'] = float(1/np.sqrt(lam)); r['observed_cv'] = float(counts.std()/mean)
    # split-half by document: choose static top on half A, evaluate hit rate on half B
    ud = np.unique(docs); A = np.isin(docs, ud[0::2]); B = ~A
    cA = np.bincount(ids[A].ravel(), minlength=E); order = np.argsort(cA)[::-1]
    cB = np.bincount(ids[B].ravel(), minlength=E); orderB = np.argsort(cB)[::-1]
    r['static_top_heldout_hit'] = {C: static_hit(ids[B], order[:C]) for C in sizes}
    r['static_top_oracle_hit'] = {C: static_hit(ids[B], orderB[:C]) for C in sizes}
    r['uniform_hit'] = {C: C/E for C in sizes}
    if extra_split is not None:
        a, b, nm = extra_split
        ca = np.bincount(ids[a].ravel(), minlength=E); o = np.argsort(ca)[::-1]
        cb = np.bincount(ids[b].ravel(), minlength=E); ob = np.argsort(cb)[::-1]
        r[f'static_{nm}_hit'] = {C: static_hit(ids[b], o[:C]) for C in sizes}
        r[f'static_{nm}_oracle'] = {C: static_hit(ids[b], ob[:C]) for C in sizes}
    # temporal reuse: fraction of token t's experts present in token t-1 (same doc); and in last W tokens
    same = docs[1:] == docs[:-1]
    prev_overlap = np.array([len(set(ids[i]) & set(ids[i-1])) for i in range(1, T)])[same] / 8
    r['reuse_prev_token'] = float(prev_overlap.mean())
    r['reuse_uniform_expect'] = 8/E
    for W in [4, 16, 64]:
        vals = []
        for i in range(W, T, 7):
            if docs[i-W] != docs[i]: continue
            win = set(ids[i-W:i].ravel().tolist()); vals.append(np.mean([e in win for e in ids[i]]))
        r[f'reuse_last{W}'] = float(np.mean(vals)); r[f'reuse_last{W}_uniform'] = float(1-(1-8/E)**W)
    # unique experts per window of B consecutive tokens (MTP verify batch)
    for Bt in [2, 4, 8, 32]:
        u = [len(np.unique(ids[i:i+Bt])) for i in range(0, T-Bt, Bt) if docs[i] == docs[i+Bt-1]]
        r[f'unique_per_{Bt}tok'] = float(np.mean(u))
    r['lru_hit'] = lru_sim(ids, docs, E, sizes)
    r['lru_plus_static_warm'] = lru_sim(ids[B], docs[B], E, sizes, static=order)
    return r

res = []
sizes_glm = [16, 32, 64, 96, 128]
# GLM matched-context (control + ood documents), 4 layers
for L in [16, 32, 49, 66]:
    ids, p, docs, pos, domains = load_glm(f'{OD}/glm53_matched_context_pilot_v1_capture/layer_{L}.pt')
    # domain per document
    dom = np.array(domains)[docs]
    ctl = np.array([str(x).startswith('control') for x in dom]); extra = (ctl, ~ctl, 'control_to_ood')
    r = analyse(ids, p, docs, 256, f'GLM L{L} matched(ctl+ood)', sizes_glm, extra); res.append(r)
    print(json.dumps(r), flush=True)
for L in [16, 22, 25, 28, 32, 38, 49, 53, 55]:
    ids, p, docs, pos, domains = load_glm(f'{OD}/native_id_control_v1_capture/layer_{L}.pt')
    r = analyse(ids, p, docs, 384, f'MiMo L{L} native-id control', [24, 48, 96, 144, 192]); res.append(r); print(json.dumps(r), flush=True)
# MiMo layers 65-67: 8 ranks x 12288 tokens
sizes_mi = [24, 48, 96, 144, 192]
for L in [65, 66, 67]:
    I = []; P = []; S = []
    for rk in range(8):
        d = torch.load(f'{MI}/capture{L}/rank{rk}.pt', map_location='cpu', mmap=True, weights_only=False)
        I.append(d['topk_ids'].numpy().copy()); P.append(d['topk_weights'].numpy().copy()); S.append(d['sequence_ids'].numpy().copy() + rk * 10**7)
        del d
    ids = np.concatenate(I); p = np.concatenate(P); docs = np.concatenate(S)
    r = analyse(ids, p, docs, 384, f'MiMo L{L} calib', sizes_mi); res.append(r); print(json.dumps(r), flush=True)
json.dump(res, open('/home/coder/git/nestquant/threads/10-streaming-system/router_stats.json', 'w'), indent=1)
