"""Level-2 analogue: fractional per-projection base rate (gate/up 1.75 = 1.5/2 mix, down 2.5) -> 2.0 average.
Placement positional (higher rate on last half in LDL order) vs alternate (benefit-blind).  Kr = 2 uniform."""
import sys, json
from alloc_lib import *
L, E = int(sys.argv[1]), int(sys.argv[2])
data, Hs = load(L, E)
D = f'{SCR}/L{L}E{E}'; rec_path = f'results/compose_L{L}E{E}.json'; rec = json.load(open(rec_path)) if os.path.exists(rec_path) else {}
for mi, pn in enumerate(PROJ):
    P = problem(data.teacher[mi], Hs[mi], mi)
    m, n = P.Wn.shape; nc, nr = n // U, m // U; sh = shard_of(nc, nr, mi == 2)
    two = torch.full((nc, nr), 2.0)
    if mi == 2:
        plans = {'uniform': torch.full((nc, nr), 2.5)}
    else:
        pos = torch.full((nc, nr), 1.5); pos[nc // 2:] = 2.0
        alt = torch.full((nc, nr), 1.5); alt[1::2] = 2.0
        plans = {'positional': pos, 'alternate': alt}
    for tag, Kb in plans.items():
        o = fit(P, 0.3, Kb, two); t = f'B{float(Kb.mean())}_{tag}'; save_w(P, o, f'{D}/{pn}/{t}.pt')
        rec[f'{pn}/{t}'] = dict(l2=o['l2'], l4=o['l4'], bpw=bits(m, n, Kb, two, mi == 2)); print(pn, t, rec[f'{pn}/{t}'], flush=True)
        json.dump(rec, open(rec_path, 'w'), indent=1)
    del P; torch.cuda.empty_cache()
