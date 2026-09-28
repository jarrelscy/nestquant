"""Candidate 3a: per-projection rate allocation (base and residual), bpw-neutral: mean over the three
equal-size projections of Kb = 2.0 and of Kr = 2.0. Non-grid averages via per-16-col-block K lists
(evenly interleaved half-bit steps). Fits cached per (proj, lam, Kb, Kr)."""
import sys, json, itertools
from t14 import *
from orbit_duet.source import weights
L, E = int(sys.argv[1]), int(sys.argv[2])
CFG = sys.argv[3] if len(sys.argv) > 3 else 'screen'
Ws = weights('/tmp/nestquant/glm53-fp8-experts', L, E)
Hs = hessians(L, E, Ws)

def klist(avg, nblk):
    """Per-block K list with the given average using the two nearest half-bit rates, evenly interleaved."""
    lo = math.floor(avg * 2) / 2; hi = lo + 0.5
    if abs(avg - lo) < 1e-9: return lo
    nh = round((avg - lo) / 0.5 * nblk)
    ks = [lo] * nblk
    for t in range(nh): ks[int((t + 0.5) * nblk / nh)] = hi
    return [k if k != int(k) else int(k) for k in ks]

def key(p, lam, kb, kr): return f'{SCR}/fit_l{L}_e{E}_{p}_lam{lam}_kb{kb}_kr{kr}.pt'

def get(mi, lam, kb, kr, P=[None]):
    f = key(PROJ[mi], lam, kb, kr)
    if os.path.exists(f): return torch.load(f)
    if P[0] is None or P[0][0] != mi:
        P[0] = None; torch.cuda.empty_cache(); P[0] = (mi, problem(Ws[mi], Hs[mi], mi))
    Pm = P[0][1]; nblk = Pm.Wn.shape[1] // 16
    o = fit(Pm, lam=lam, Kb=klist(kb, nblk), Kr=klist(kr, nblk))
    r = dict(w2=Pm.dequant(o['Q2']).bfloat16().cpu(), w4=Pm.dequant(o['Q4']).bfloat16().cpu(), l2=o['l2'], l4=o['l4'])
    torch.save(r, f); print(PROJ[mi], lam, kb, kr, f"proxy l2 {o['l2']:.5f} l4 {o['l4']:.5f}", flush=True)
    return r

if CFG == 'screen':
    # (gate/up K, down K) pairs with (2*x + d)/3 = 2
    pairs = [(2, 2), (1.875, 2.25), (1.75, 2.5), (1.625, 2.75), (1.5, 3)]
    lams = [0.3, 0.5, 0.7]
    todo = [(lam, bp, rp) for lam in lams for bp in pairs[:4] for rp in pairs]
else:
    todo = json.loads(CFG)
for mi in (2, 0, 1):
    for lam, bp, rp in todo:
        kb = bp[0] if mi < 2 else bp[1]; kr = rp[0] if mi < 2 else rp[1]
        get(mi, lam, kb, kr)
meth = {}
for lam, bp, rp in todo:
    parts = [get(mi, lam, bp[0] if mi < 2 else bp[1], rp[0] if mi < 2 else rp[1]) for mi in range(3)]
    meth[f'lam{lam}_b{bp[0]}/{bp[1]}_r{rp[0]}/{rp[1]}@4'] = [p['w4'].cuda().float() for p in parts]
    meth.setdefault(f'lam{lam}_b{bp[0]}/{bp[1]}@2', [p['w2'].cuda().float() for p in parts])
res = {}
names = list(meth)
for a in range(0, len(names), 12):
    res.update(evaluate(L, E, {k: meth[k] for k in names[a:a+12]}))
for k in sorted(res): print(f"{k:36s} routed {res[k]['routed']:7.3f} forced {res[k]['forced']:7.3f} ood {res[k]['ood']:7.3f}")
json.dump(res, open(f'results_alloc_{CFG if CFG=="screen" else "custom"}_l{L}_e{E}.json', 'w'), indent=1)
