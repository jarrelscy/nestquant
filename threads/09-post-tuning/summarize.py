"""Select lr per (model,bits,param-set) by held-out split only; tabulate eval before/after."""
import json, glob, re, collections, sys
rows = collections.defaultdict(list)
for f in sorted(glob.glob('results/*.json')):
    r = json.load(open(f))
    name = f.split('/')[-1][:-5]
    key = re.sub(r'_lr[0-9e.-]+$', '', name)
    rows[key].append((r, name))
KEYS = {'glms03': ['pilot/all/forced', 'pilot/all/routed', 'pilot/control/forced', 'pilot/ood/forced'], 'glm': ['pilot/all/forced', 'pilot/all/routed', 'pilot/control/forced', 'pilot/ood/forced'],
        'glm90': ['pilot/all/forced', 'pilot/all/routed', 'pilot/control/forced', 'pilot/ood/forced'],
        'mimo': ['control/all/forced', 'control/all/routed', 'ood/all/forced']}
out = {}
print('| run | +bpw | lr | ep | train | held-out (base->tuned) | ' + ' | '.join(['eval']*4) + ' |')
for key in sorted(rows):
    cands = rows[key]
    r, name = min(cands, key=lambda t: t[0]['best'].get('hold_rounded', t[0]['best']['hold']))
    m = r['model']
    ev = r.get('eval', {})
    cells = []
    for k in KEYS[m]:
        if k in ev:
            b, t = ev[k]['base'], ev[k]['tuned']
            cells.append(f"{k.split('/',1)[1]} {b:.2f}->{t:.2f} ({100*(t-b)/b:+.2f}%)")
    hb = r['init']['hold']; ht = r['best'].get('hold_rounded', r['best']['hold'])
    print(f"| {key} | {r['extra_bpw']:.4f} | {r['lr']:g} | {r['best']['epoch']} | {r['init']['train']:.2f}->{r['best'].get('train', r['init']['train']):.2f} | {hb:.3f}->{ht:.3f} ({100*(ht-hb)/hb:+.2f}%) | " + ' | '.join(cells) + ' |')
    out[key] = dict(selected=name, extra_bpw=r['extra_bpw'], lr=r['lr'], epoch=r['best']['epoch'], hold=(hb, ht),
                    train=(r['init']['train'], r['best'].get('train')), eval={k: (ev[k]['base'], ev[k]['tuned'], ev[k]['rows']) for k in ev})
json.dump(out, open('summary.json', 'w'), indent=1)
