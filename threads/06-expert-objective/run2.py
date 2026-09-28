"""Objective grid with training-only holdout selection.
mode=holdout: fit on training rows minus a seeded 20% of the retained training sample, score on those held-out rows.
mode=full:    refit on all training rows, score on the frozen evaluation captures (final scoring only)."""
import argparse, json, time
from pathlib import Path
import torch
import torch.nn.functional as F
import eo

p = argparse.ArgumentParser()
p.add_argument('--model', required=True); p.add_argument('--bits', type=int, nargs='+', default=[2, 4])
p.add_argument('--mode', choices=['holdout', 'full'], required=True)
p.add_argument('--configs', nargs='+', required=True)
a = p.parse_args()
eo.setup()
T = Path(f'/tmp/nestquant/06-expert-objective/{a.model}_{a.mode}'); T.mkdir(parents=True, exist_ok=True)
OUT = Path(__file__).parent / f'grid_{a.model}_{a.mode}.json'
res = json.loads(OUT.read_text()) if OUT.exists() else {}
D = eo.Data(a.model); g, u, d = D.w
t0 = time.time()
def log(*x): print(f'[{time.time()-t0:7.1f}s]', *x, flush=True)

n = len(D.sx); perm = torch.randperm(n, generator=torch.Generator().manual_seed(606))
val = perm[:n // 5].sort().values if a.mode == 'holdout' else torch.arange(0)
fit = perm[n // 5:].sort().values if a.mode == 'holdout' else torch.arange(n)
FULL_ROWS = D.model == 'mimo'  # MiMo stats cover 408k rows; sample is a 32k subset


def rows(idx, bs=2048):
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]; yield D.sx[j].cuda(), D.sp[j].cuda()


@torch.no_grad()
def grams(idx, pw, gq=None, uq=None):
    Z = lambda k: torch.zeros(k, k, device='cuda', dtype=torch.float32)
    o = dict(Hx=Z(6144), Ha=Z(2048), Og=Z(2048), Ou=Z(2048), m=0.)
    if gq is not None: o.update(Hq=Z(2048), C=Z(2048))
    for x, pp in rows(idx):
        r = pp.pow(pw / 2)[:, None]; xf = x.float()
        o['Hx'].addmm_((xf * r).T, xf * r)
        a_ = D.hidden(x, g, u) * r; o['Ha'].addmm_(a_.T, a_)
        gx, ux = F.linear(xf, g), F.linear(xf, u); sig = gx.sigmoid()
        dg = ux * sig * (1 + gx * (1 - sig)) * r; du = F.silu(gx) * r
        o['Og'].addmm_(dg.T, dg); o['Ou'].addmm_(du.T, du); o['m'] += float(r.square().sum())
        if gq is not None:
            aq = D.hidden(x, gq, uq) * r; o['Hq'].addmm_(aq.T, aq); o['C'].addmm_(a_.T, aq)
    return o


_base = {}
def base_grams(pw, src='auto', gb='1'):
    """Fit-set grams. MiMo pw=2 uses full statistics minus held-out sample rows; otherwise sample fit rows."""
    if (pw, src) in _base: return _base[(pw, src)]
    if FULL_ROWS and pw == 2 and src == 'auto':
        o = dict(Hx=D.grams[0].cuda().clone(), Ha=D.grams[1].cuda().clone(), Og=D.outputs[0].cuda().clone(), Ou=D.outputs[1].cuda().clone(), m=D.meta['mass'])
        if len(val):
            v = grams(val, 2)
            for k in ['Hx', 'Ha', 'Og', 'Ou']: o[k] -= v[k]
            o['m'] -= v['m']
        o['count'] = D.count - len(val)
    else:
        o = grams(fit, pw); o['count'] = len(fit)
    _base[(pw, src)] = o; return o


def parse(c):
    kv = dict(gu='H', dn='H', pw='2', sr='0.03', ridge='1e-3', src='auto', gb='1')
    for part in c.split(','):
        k, v = part.split('='); kv[k] = v
    return kv


def cached(key, fn):
    f = T / f'{key}.pt'
    if f.exists(): return torch.load(f).cuda()
    w, proxy = fn(); torch.save(w.cpu(), f); log(key, 'proxy %.6f' % proxy); return w


@torch.no_grad()
def score_rows(wq, idx):
    num = den = numu = denu = 0.
    for x, pp in rows(idx):
        t = eo.teacher(x, D.w).double(); e = (eo.teacher(x, wq).double() - t).square().sum(-1); en = t.square().sum(-1)
        q = pp.double().square(); num += float((e * q).sum()); den += float((en * q).sum()); numu += float(e.sum()); denu += float(en.sum())
    return dict(val_p2=(num / den) ** .5, val_uniform=(numu / denu) ** .5)


M = d.T @ d
for bits in a.bits:
    for c in a.configs:
        key = f'{c}@{bits}'
        if key in res: log('skip', key); continue
        kv = parse(c); pw = float(kv['pw']); sr = float(kv['sr']); B = base_grams(pw, kv['src'])
        extra = dict(sigma_reg=sr)
        tag = f"b{bits}_pw{kv['pw']}_sr{kv['sr']}_{kv['src']}"
        Wq = []
        for i, (nm, W) in enumerate([('g', g), ('u', u)]):
            O = B['Og'] if i == 0 else B['Ou']
            Hout = {'H': lambda: None, 'Gfull': lambda: M * O, 'Gdiag': lambda: torch.diag((M * O).diagonal()),
                    'Gid': lambda: torch.eye(2048, device='cuda'), 'GMdiag': lambda: torch.diag(M.diagonal()),
                    'GOdiag': lambda: torch.diag(O.diagonal().clamp_min(1e-30).pow(float(kv['gb']))), 'GM': lambda: M.clone()}[kv['gu']]()
            Wq.append(cached(f"{nm}_{tag}_{kv['gu']}" + ('' if kv['gb'] == '1' else f"_gb{kv['gb']}"), lambda W=W, Hout=Hout: eo.quant(W, B['Hx'], B['count'], bits, H_out=Hout, **extra)))
        gq, uq = Wq
        if kv['dn'] == 'H':
            dq = cached(f'd_{tag}_H', lambda: eo.quant(d, B['Ha'], B['count'], bits, **extra))
        else:
            S = grams(fit, pw, gq, uq); s = B['m'] / S['m']
            Hq = B['Ha'] + s * (S['Hq'] - S['Ha']); C = B['Ha'] + s * (S['C'] - S['Ha'])
            if kv['dn'] == 'seqH':
                dq = cached(f"d_{tag}_seqH_{kv['gu']}", lambda: eo.quant(d, Hq, B['count'], bits, **extra))
            else:
                # ridge shrinks the regression target toward the teacher W_d (not toward 0)
                lam = float(kv['ridge']) * float(Hq.diagonal().mean())
                R = Hq.double().clone(); R.diagonal().add_(lam)
                Ws = (d.double() @ (C.double() + lam * torch.eye(2048, device='cuda', dtype=torch.float64)) @ torch.linalg.inv(R)).float()
                dq = cached(f"d_{tag}_seqW_{kv['gu']}_r{kv['ridge']}" + ('' if kv['gb'] == '1' else f"_gb{kv['gb']}"), lambda: eo.quant(Ws, Hq, B['count'], bits, **extra))
            del S
        wq = [gq, uq, dq]
        r = dict(score_rows(wq, val) if len(val) else {}, train_fit_p2=score_rows(wq, fit[:8192])['val_p2'])
        if a.mode == 'full': r['eval'] = {k: v['rel'] for k, v in D.evaluate(wq).items()}
        res[key] = r; OUT.write_text(json.dumps(res, indent=1))
        log(key, {k: (round(v * 100, 2) if isinstance(v, float) else {kk: round(vv * 100, 2) for kk, vv in v.items()}) for k, v in r.items()})
        torch.cuda.empty_cache()
