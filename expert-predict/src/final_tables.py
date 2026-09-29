import json, collections, sys
R = '/data/Jarrel/expert-predict/results/'
VAL = ['formal-crypto', 'embedding-drift-monitor']; CLEAN = ['freight-dispatch-shift', 'sound-change-cascade']; SW = ['fin-saccr-rwa', 'pretrain-shard-corruption']
rows = collections.defaultdict(dict); Ntok = {}
for l in open(R + 'cap_curve.jsonl'):
    r = json.loads(l); k = r['pred'] + ('_lazy' if r['lazy'] else ''); rows[(k, r['cap'])][r['task']] = r; Ntok[r['task']] = r['N']
for f in ('gbdt_eval.jsonl',):
    for l in open(R + f):
        r = json.loads(l)
        if r['task'].startswith('probe') or 'recall_gbdt' not in r: continue
        k = f"gbdt_{r['tgt']}_hm{r['hm']}"; rows[(k, r['cap'])][r['task']] = r; Ntok[r['task']] = r['N']
def wmean(d, ts, key='share'):
    ts = [t for t in ts if t in d]; n = sum(Ntok[t] for t in ts); return sum(d[t][key] * Ntok[t] for t in ts) / n if n else float('nan')
cap = float(sys.argv[1]) if len(sys.argv) > 1 else 24.0
print(f'cap {cap} aggregate GB/s: val(formal,embed) | freight sound fin pretrain | clean-mean | GB/s(clean)')
out = []
for (k, c), d in rows.items():
    if c != cap: continue
    v = wmean(d, VAL); cl = wmean(d, CLEAN); g = wmean(d, CLEAN, 'gbps')
    out.append((v, k, ' '.join('%.4f' % d[t]['share'] if t in d else '  -   ' for t in CLEAN + SW), cl, g))
for v, k, s, cl, g in sorted(out, key=lambda x: -x[0] if x[0] == x[0] else 0):
    print('%-34s %.4f | %s | %.4f | %.1f' % (k, v, s, cl, g))
