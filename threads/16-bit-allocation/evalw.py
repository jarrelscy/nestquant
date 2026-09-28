"""Score saved dequantized weights with harness.evaluate on the matched capture.
usage: evalw.py L E out.json tag1 tag2 ...   (tag = <file stem> present for all three projections, or
 'gate=a,up=b,down=c' mixes).  Each tag yields tag@2 and tag@4 (w2 / w4).  Adds EXL3 anchors if present."""
import sys, json, torch
from alloc_lib import *
L, E, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
data = hh.load_expert(L, E)
D = f'{SCR}/L{L}E{E}'
def parts(tag):
    if '=' in tag:
        mp = dict(x.split('=') for x in tag.split(','))
        return [torch.load(f'{D}/{p}/{mp[p]}.pt') for p in PROJ]
    return [torch.load(f'{D}/{p}/{tag}.pt') for p in PROJ]
methods = {}
for tag in sys.argv[4:]:
    ps = parts(tag)
    if tag.startswith('EXL3'):
        methods[tag] = [p['w4'].cuda() for p in ps]
    else:
        methods[tag + '@2'] = [p['w2'].cuda() for p in ps]; methods[tag + '@4'] = [p['w4'].cuda() for p in ps]
tab = hh.table(hh.evaluate(data, methods))
res = json.load(open(out)) if os.path.exists(out) else {}
for k, t in tab.items():
    res[k] = t
    print(f'{k:40s} routed {t["all/routed"]:7.3f} forced {t["all/forced"]:7.3f} ood {t["ood/forced"]:7.3f} oodR {t.get("ood/routed", float("nan")):7.3f}', flush=True)
json.dump(res, open(out, 'w'), indent=1)
