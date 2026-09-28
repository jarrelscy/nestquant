"""Composition with thread-14 per-projection fractional K: a projection average between supported rates forces a
within-projection mix.  Compare placements at the same bytes/shard: greedy (benefit per bit), alternate
(checkerboard, benefit-blind), anti-greedy (worst).  Refinement plane only (base = K2)."""
import sys, json
from alloc_lib import *
L, E = int(sys.argv[1]), int(sys.argv[2])
data, Hs = load(L, E)
D = f'{SCR}/L{L}E{E}'; rec_path = f'results/compose_L{L}E{E}.json'; rec = json.load(open(rec_path)) if os.path.exists(rec_path) else {}
CASES = {'gate': [1.75], 'up': [1.75], 'down': [2.5]} if len(sys.argv) > 3 else {'gate': [1.75, 2.25], 'up': [1.75, 2.25], 'down': [2.25, 2.5]}
for mi, pn in enumerate(PROJ):
    P = problem(data.teacher[mi], Hs[mi], mi)
    m, n = P.Wn.shape; nc, nr = n // U, m // U; sh = shard_of(nc, nr, mi == 2)
    cur = torch.load(f'{D}/{pn}/curves_uni.pt' if os.path.exists(f'{D}/{pn}/curves_uni.pt') else f'{D}/{pn}/curves_none.pt')['c4']
    two = torch.full((nc, nr), 2.0)
    for R in CASES[pn]:
        lo, hi = (1.5, 2.0) if R < 2 else ((2.0, 2.5) if R < 2.5 else (2.5, 2.5))
        if lo == hi:
            plans = {'uniform': torch.full((nc, nr), R)}
        else:
            g = allocate({k: cur[k] for k in (lo, hi)}, R, sh, (lo, hi))
            ben = cur[lo] - cur[hi]; worst = torch.full((nc, nr), lo); alt = torch.full((nc, nr), lo)
            for s in sh.unique():
                idx = (sh == s).nonzero(); k = int(round(len(idx) * (R - lo) / (hi - lo)))
                o_ = torch.argsort(ben[idx[:, 0], idx[:, 1]]); w = idx[o_[:k]]; worst[w[:, 0], w[:, 1]] = hi
                a = idx[torch.linspace(0, len(idx) - 1, k).round().long()]; alt[a[:, 0], a[:, 1]] = hi
            pos = torch.full((nc, nr), lo); pos[nc - int(round(nc * (R - lo) / (hi - lo))):] = hi   # last chunks in LDL order
            plans = {'greedy': g, 'alternate': alt, 'antigreedy': worst, 'positional': pos}
            if len(sys.argv) > 4: plans = {k: plans[k] for k in sys.argv[4].split(',')}
        for tag, Kr in plans.items():
            pass
        for tag, Kr in plans.items():
            o = fit(P, 0.3, two, Kr)
            t = f'C{R}_{tag}'; save_w(P, o, f'{D}/{pn}/{t}.pt')
            rec[f'{pn}/{t}'] = dict(l2=o['l2'], l4=o['l4'], bpw=bits(m, n, two, Kr, mi == 2), mean_r=float(Kr.mean()))
            print(pn, t, rec[f'{pn}/{t}'], flush=True)
            json.dump(rec, open(rec_path, 'w'), indent=1)
    del P; torch.cuda.empty_cache()

# skip-refinement variant at exactly 2.0 mean (4.0 total): 1/6 of units per shard carry no P4, 1/6 at K2, 2/3 at K2.5
for mi, pn in enumerate(PROJ):
    P = problem(data.teacher[mi], Hs[mi], mi)
    m, n = P.Wn.shape; nc, nr = n // U, m // U; sh = shard_of(nc, nr, mi == 2)
    cur = torch.load(f'{D}/{pn}/curves_uni.pt' if os.path.exists(f'{D}/{pn}/curves_uni.pt') else f'{D}/{pn}/curves_none.pt')['c4']
    two = torch.full((nc, nr), 2.0); Kr = two.clone()
    for s in sh.unique():
        idx = (sh == s).nonzero(); c = cur[2][idx[:, 0], idx[:, 1]]; o_ = torch.argsort(c)   # ascending unit cost
        k = len(idx) // 6; z = idx[o_[:k]]; hi = idx[o_[2 * k:]]
        Kr[z[:, 0], z[:, 1]] = 0; Kr[hi[:, 0], hi[:, 1]] = 2.5
    o = fit(P, 0.3, two, Kr); t = 'S0_skip'; save_w(P, o, f'{D}/{pn}/{t}.pt')
    rec[f'{pn}/{t}'] = dict(l2=o['l2'], l4=o['l4'], bpw=bits(m, n, two, Kr, mi == 2), mean_r=float(Kr.mean()))
    print(pn, t, rec[f'{pn}/{t}'], flush=True); json.dump(rec, open(rec_path, 'w'), indent=1)
    del P; torch.cuda.empty_cache()
