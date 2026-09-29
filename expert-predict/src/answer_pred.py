"""State-aware score: think state -> global EMA(hl); answer state -> (1-a-b)*global + a*own-clock answer EMA + b*answer prior.
Also optional think-state own-clock memory (c) so the think profile is restored when a new request starts."""
import sys, json, itertools; sys.path.insert(0, '/data/Jarrel/expert-predict/src'); from feats import *
def scores(P, R, hl, shl, a, b, c, prior):
    C, Cth, Can, bst, nans = P['C'], P['Cth'], P['Can'], P['bst'], P['nans']
    nb = C.shape[0]; step = R // G; nref = -(-nb // step) + 1
    ag = np.float32(0.5 ** (G / hl)); sa = 0.5 ** (1 / shl)
    E = np.zeros((NL, NE), np.float32); Et = np.zeros_like(E); Ea = np.zeros_like(E); wt = wa = 0.0
    S = np.zeros((nref, NL, NE), np.float32); ng = (1 - ag) / G
    for bb in range(nb + 1):
        if bb % step == 0:
            s = int(bst[bb - 1]) if bb > 0 else 0; g = E * ng
            if s == 1: S[bb // step] = (1 - a - b) * g + a * Ea / max(wa, 1e-6) + b * prior[1]
            else: S[bb // step] = (1 - c) * g + c * Et / max(wt, 1e-6)
        if bb == nb: break
        E = E * ag + C[bb]; na = nans[bb]; nt = G - na
        dt = np.float32(sa ** nt); da = np.float32(sa ** na)
        Et = Et * dt + Cth[bb]; wt = wt * dt + nt; Ea = Ea * da + Can[bb]; wa = wa * da + na
    return S
if __name__ == '__main__':
    tasks = sys.argv[1].split(','); train = sys.argv[2].split(','); out = open(sys.argv[3], 'a')
    prior = priors(train)
    grid = json.loads(sys.argv[4])  # list of dicts
    for t in tasks:
        d = load(t); P = prep(d); st = P['st']
        for g in grid:
            S = scores(P, g['R'], g['hl'], g.get('shl', 2048), g.get('a', 0), g.get('b', 0), g.get('c', 0), prior)
            for hm in g.get('hms', [0.0]):
                for lazy in g.get('lazys', [0]):
                    o = sim(d['ex'], S, R=g['R'], hm=hm, lazy=lazy, lead=g.get('lead', 13), cap_gbps=g.get('cap', 6.0)); h = o.pop('hits') / 600.
                    rec = dict(task=t, **{k: v for k, v in g.items() if k not in ('hms', 'lazys')}, hm=hm, lazy=lazy,
                               share=o['share'], think=float(h[st == 0].mean()), answer=float(h[st == 1].mean()),
                               gbps=o['gbps'], sal=o['sal_share'], N=o['N'])
                    print(json.dumps(rec), flush=True); out.write(json.dumps(rec) + '\n'); out.flush()
