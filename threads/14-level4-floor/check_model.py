from t14 import *
S = SCR; f = lambda p, kb, kr=2: torch.load(f'{S}/fit_l16_e36_{p}_lam0.3_kb{kb}_kr{kr}.pt')
g, u = f('gate', 2), f('up', 2)
m = {}
for kd in (2, 2.25, 2.5, 2.75):
    d = f('down', kd); m[f'down{kd}@2'] = [g['w2'], u['w2'], d['w2']]
    m[f'down_kb{kd}@4'] = [g['w4'], u['w4'], d['w4']]
for kg in (1.75, 1.875):
    m[f'gate{kg}@2'] = [f('gate', kg)['w2'], u['w2'], f('down', 2)['w2']]
r = evaluate(16, 36, {k: [w.cuda().float() for w in v] for k, v in m.items()})
for k, v in r.items(): print(k, v)
