import json, os, glob, torch
EXP = [(L, E) for L in (16, 49, 66) for E in (36, 92, 165)]
M = [('EXL3-2', 'EXL3_2'), ('uniform 2+2 @2', 'uni_none@2'), ('greedy P4 (A4g) @2', 'gate=A4g_none,up=A4g_none,down=uni_none@2'),
     ('forced +-0.5 base (A2pm) @2', 'A2pm_none@2'),
     ('EXL3-4', 'EXL3_4'), ('uniform 2+2 @4', 'uni_none@4'), ('greedy P4 (A4g) @4', 'gate=A4g_none,up=A4g_none,down=uni_none@4'),
     ('forced +-0.5 P4 (A4pm) @4', 'A4pm_none@4'),
     ('g/u 1.75 greedy, d 2.5 @4', 'gate=C1.75_greedy,up=C1.75_greedy,down=C2.5_uniform@4'),
     ('g/u 1.75 positional, d 2.5 @4', 'gate=C1.75_positional,up=C1.75_positional,down=C2.5_uniform@4'),
     ('g/u 1.75 alternate, d 2.5 @4', 'gate=C1.75_alternate,up=C1.75_alternate,down=C2.5_uniform@4')]
COLS = ['all/routed', 'all/forced', 'ood/forced', 'ood/routed']
R = {}
for L, E in EXP:
    f = 'results/e36_screen.json' if (L, E) == (16, 36) else f'results/eval_L{L}E{E}.json'
    if os.path.exists(f): R[(L, E)] = json.load(open(f))
out = {}
print('| method | ' + ' | '.join(COLS) + ' | per-expert routed (L16 36/92/165, L49 ..., L66 ...) |')
for name, key in M:
    have = [e for e in EXP if e in R and key in R[e]]
    if not have: continue
    mean = {c: sum(R[e][key][c] for e in have) / len(have) for c in COLS}
    out[name] = dict(n=len(have), mean=mean, per={f'L{e[0]}E{e[1]}': R[e][key] for e in have})
    print(f'| {name} (n={len(have)}) | ' + ' | '.join(f'{mean[c]:.3f}' for c in COLS) + ' | ' + ' '.join(f'{R[e][key]["all/routed"]:.2f}' for e in have) + ' |')
json.dump(out, open('results/nine_expert_summary.json', 'w'), indent=1)
# win counts vs uniform
def wins(key, base, c):
    have = [e for e in EXP if e in R and key in R[e] and base in R[e]]
    return sum(R[e][key][c] < R[e][base][c] for e in have), len(have)
for name, key in M:
    if key.endswith('@4') and key != 'uni_none@4':
        print(name, 'beats uniform@4:', {c: wins(key, 'uni_none@4', c) for c in COLS})
    if key.endswith('@2') and key != 'uni_none@2':
        print(name, 'beats uniform@2:', {c: wins(key, 'uni_none@2', c) for c in COLS})
print('positional beats alternate:', {c: wins('gate=C1.75_positional,up=C1.75_positional,down=C2.5_uniform@4', 'gate=C1.75_alternate,up=C1.75_alternate,down=C2.5_uniform@4', c) for c in COLS})
# ideal continuous-rate AM/GM bound of unit costs
amgm = lambda x: float(10 * torch.log10(x.mean() / x.log().mean().exp()))
S = '/tmp/nestquant/16-bit-allocation'
for p in ('gate', 'up', 'down'):
    v2, v4 = [], []
    for L, E in EXP:
        f = f'{S}/L{L}E{E}/{p}/curves_none.pt'
        if not os.path.exists(f): f = f'{S}/L{L}E{E}/{p}/curves_uni.pt'
        c = torch.load(f); v2.append(amgm(c['c2'][2].double().flatten())); v4.append(amgm(c['c4'][2].double().flatten()))
    print(p, 'unit-cost AM/GM dB (ideal continuous allocation bound) L2 %.3f-%.3f  L4 %.3f-%.3f' % (min(v2), max(v2), min(v4), max(v4)))
