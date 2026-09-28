import argparse, json, time, os
from pathlib import Path
import torch
import eo

p = argparse.ArgumentParser()
p.add_argument('--model', required=True); p.add_argument('--bits', type=int, nargs='+', default=[2, 4])
p.add_argument('--variants', nargs='+', default=['stored', 'base', 'seqH', 'seq', 'Gdiag', 'Gfull', 'Gfull+seq', 'pw0', 'pw2s'])
p.add_argument('--ridge', type=float, default=1e-3)
a = p.parse_args()
eo.setup()
T = Path(f'/tmp/nestquant/06-expert-objective/{a.model}'); T.mkdir(parents=True, exist_ok=True)
OUT = Path(__file__).parent / f'results_{a.model}.json'
res = json.loads(OUT.read_text()) if OUT.exists() else {}
D = eo.Data(a.model); g, u, d = D.w
t0 = time.time()
def log(*x): print(f'[{time.time()-t0:7.1f}s]', *x, flush=True)

M = d.T @ d  # (2048,2048) down sensitivity
Hout = dict(Gfull=[M * D.outputs[0].cuda(), M * D.outputs[1].cuda()])
Hout['Gdiag'] = [torch.diag(h.diagonal()) for h in Hout['Gfull']]
spread = {k: [eo.rotated_spread(h) for h in v] for k, v in Hout.items()}
res['G_spread'] = spread; log('spread', json.dumps(spread))

_sg = {}
def sgrams(pw):
    if pw not in _sg: _sg[pw] = D.sample_grams(pw)
    return _sg[pw]

def cached(key, fn):
    f = T / f'{key}.pt'
    if f.exists(): return torch.load(f).cuda()
    w, proxy = fn(); torch.save(w.cpu(), f); log(key, 'proxy', proxy); return w

def gate_up(bits, obj):
    if obj == 'base':
        return [cached(f'{n}_{bits}_base', lambda W=W: eo.quant(W, D.grams[0], D.count, bits)) for n, W in [('g', g), ('u', u)]]
    if obj in ('Gfull', 'Gdiag'):
        return [cached(f'{n}_{bits}_{obj}', lambda W=W, i=i: eo.quant(W, D.grams[0], D.count, bits, H_out=Hout[obj][i])) for i, (n, W) in enumerate([('g', g), ('u', u)])]
    if obj.startswith('pw'):
        pw = float(obj[2]); S = sgrams(pw)
        return [cached(f'{n}_{bits}_{obj}', lambda W=W: eo.quant(W, S['Hx'], len(D.sx), bits)) for n, W in [('g', g), ('u', u)]]
    raise ValueError(obj)

def down(bits, obj, gq, uq, tag):
    if obj == 'base': return cached(f'd_{bits}_base', lambda: eo.quant(d, D.grams[1], D.count, bits))
    if obj.startswith('pw'):
        S = sgrams(float(obj[2])); return cached(f'd_{bits}_{obj}', lambda: eo.quant(d, S['Ha'], len(D.sx), bits))
    S = D.sample_grams(2.0, gq, uq); s = D.full_scale; G1 = D.grams[1].cuda()
    Hq = G1 + s * (S['Hq'] - S['Ha']); C = G1 + s * (S['C'] - S['Ha'])
    if obj == 'seqH': return cached(f'd_{bits}_seqH_{tag}', lambda: eo.quant(d, Hq, D.count, bits))
    if obj == 'seq':
        R = Hq.double().clone(); R.diagonal().add_(a.ridge * R.diagonal().mean())
        Wstar = (d.double() @ C.double() @ torch.linalg.inv(R)).float()
        return cached(f'd_{bits}_seq_{tag}_r{a.ridge:g}', lambda: eo.quant(Wstar, Hq, D.count, bits))
    raise ValueError(obj)

for bits in a.bits:
    for v in a.variants:
        key = f'{v}@{bits}'
        if key in res: log('skip', key); continue
        if v == 'stored':
            from orbit_duet.exl3_adapter import EXL3Expert
            wq = EXL3Expert(f"{D.c['exl3']}/expert_{bits}.bin").decoded_weights()
        else:
            gu = {'base': 'base', 'seqH': 'base', 'seq': 'base', 'Gdiag': 'Gdiag', 'Gfull': 'Gfull', 'Gfull+seq': 'Gfull', 'Gdiag+seq': 'Gdiag',
                  'pw0': 'pw0', 'pw2s': 'pw2', 'pw1': 'pw1'}[v]
            dn = {'base': 'base', 'seqH': 'seqH', 'seq': 'seq', 'Gdiag': 'base', 'Gfull': 'base', 'Gfull+seq': 'seq', 'Gdiag+seq': 'seq',
                  'pw0': 'pw0', 'pw2s': 'pw2', 'pw1': 'pw1'}[v]
            gq, uq = gate_up(bits, gu); dq = down(bits, dn, gq, uq, gu); wq = [gq, uq, dq]
        r = dict(train=D.train_error(wq), eval=D.evaluate(wq),
                 layer_rel_w=[float((x - y).norm() / y.norm()) for x, y in zip(wq, D.w)])
        res[key] = r; OUT.write_text(json.dumps(res, indent=1))
        log(key, 'train %.4f' % r['train'], {k: round(v['rel'] * 100, 2) for k, v in r['eval'].items()})
