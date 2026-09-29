"""Per-arm rows for a C2 pass: KLD mean +- se (window-level) per corpus, top-1, ups / 1k tok / layer, and the paired
per-window KLD delta vs a base arm (same tokens, same windows: the se of the difference is far below the per-arm se).
  report_arms.py TAG [BASE=dyn_gbdt] [--out /data/Jarrel/nq-eval]"""
import sys, json, math
import numpy as np

tag = sys.argv[1]; base = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else 'dyn_gbdt'
out = sys.argv[sys.argv.index('--out') + 1] if '--out' in sys.argv else '/data/Jarrel/nq-eval'
d = json.load(open(f'{out}/results/{tag}.json')); z = np.load(f'{out}/results/{tag}_tokkl.npz')
names = list(d['inputs']['corpora'].get('windows', {}) or {}) or sorted({k.split('_')[-1] for k in z.files if k.startswith('win_')})
res, tr = d['results'], d.get('traffic', {})
arms = [s for s in res if s.startswith('dyn')] + [s for s in res if not s.startswith('dyn')]
cn = [c for c in next(iter(res.values())) if isinstance(next(iter(res.values()))[c], dict) and 'kld' in next(iter(res.values()))[c]]


def per_window(s, g):
    k, w = z[f'{s}_{g}'], z[f'win_{g}']
    n = len(k) // len(w)                                   # tokens per window block (SEQ-1)
    return k.reshape(len(w), n).mean(1)


print(f'# {tag}: per-arm rows (paired delta vs {base}: mean over windows of KLD(arm) - KLD({base}), +- se)\n')
print('| arm | corpus | KLD | +-se | top-1 | d vs ' + base + ' | +-se | ups/1k tok/layer | GBDT late waits |')
print('|---|---|---|---|---|---|---|---|---|')
for s in arms:
    for g, c in enumerate(cn):
        r = res[s][c]; row = f"| {s} | {c} | {r['kld']:.5f} | {r['kld_se']:.5f} | {100 * r['top1']:.2f}% |"
        if s != base and f'{s}_{g}' in z.files and f'{base}_{g}' in z.files:
            dd = per_window(s, g) - per_window(base, g); row += f' {dd.mean():+.5f} | {dd.std() / math.sqrt(len(dd)):.5f} |'
        else: row += ' | |'
        t = dict(tr.get(s, {}))
        if t and 'ups_per_1k_tok_per_layer' not in t:     # older results: layers = per-layer rows carrying this stream's ups
            nl = sum(1 for v in d['per_layer'].values() if f'{s}_ups' in v); t['ups_per_1k_tok_per_layer'] = t['upgrades'] / t['tokens'] / max(nl, 1) * 1e3
        row += f" {t.get('ups_per_1k_tok_per_layer', float('nan')):.2f} | {t.get('gbdt_late_waits', '')} |" if t else ' | |'
        print(row)
