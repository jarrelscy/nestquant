"""Is the level-2 residual predictable from anything the decoder already has (base value, base state bits,
neighbouring base values)? Gains in dB of residual MSE before the residual stage (upper bound on what a
conditional codebook can gain at high rate: mean removal -> variance ratio; scale-per-context -> AM/GM)."""
import torch, math, json
SCR = '/tmp/nestquant/14-level4-floor'
out = {}
for p in ['gate', 'up', 'down']:
    col = torch.load(f'{SCR}/col_{p}.pt')
    r = torch.cat([c['r'] for c in col], 1).double()      # [m, n]
    q2 = torch.cat([c['q2'] for c in col], 1).double()
    i2 = torch.cat([c['i2'] for c in col], 1).long() & 0xFFFF
    rv = r.flatten(); tot = rv.var()
    res = dict(kurt=float(((rv - rv.mean())**4).mean() / tot**2))
    def ctx_gain(c, nb):
        c = c.flatten(); cnt = torch.bincount(c, minlength=nb).double().clamp_min(1)
        mu = torch.bincount(c, rv, nb) / cnt
        v = torch.bincount(c, rv**2, nb) / cnt - mu**2
        w = cnt / cnt.sum()
        mean_gain = float(tot / (w * v).sum())
        amgm = float((w * v).sum() / torch.exp((w * v.clamp_min(1e-12).log()).sum()))
        return 10 * math.log10(mean_gain), 10 * math.log10(amgm)
    qs = torch.quantile(q2.flatten()[::97].float(), torch.linspace(0, 1, 17)[1:-1]).double()
    qb = torch.bucketize(q2, qs)
    res['q2_bin16 (mean, scale) dB'] = ctx_gain(qb, 16)
    res['state_low4'] = ctx_gain(i2 & 15, 16)
    res['state_top4'] = ctx_gain(i2 >> 12, 16)
    res['state_low8'] = ctx_gain(i2 & 255, 256)
    # linear prediction from neighbouring base values along the tile (rows), +-3
    m, n = q2.shape
    T = q2.T.reshape(-1, 256); R = r.T.reshape(-1, 256)
    feats = [torch.roll(T, s, 1) for s in range(-3, 4)] + [T.abs(), T**2]
    X = torch.stack([f.flatten() for f in feats], 1); y = R.flatten()
    X = torch.cat([X, torch.ones(len(y), 1, dtype=X.dtype)], 1)
    beta = torch.linalg.lstsq(X[::7], y[::7, None]).solution
    ev = y - (X @ beta).squeeze(1)
    res['linear_nbr+-3 dB'] = 10 * math.log10(float(y.var() / ev.var()))
    out[p] = res; print(p, res, flush=True)
json.dump(out, open('results_cond_diag.json', 'w'), indent=1)
