"""Fit per-layer (x per-state) linear future-rate models on train tasks, evaluate in the scheduler sim on eval tasks.
usage: fit_linear.py TRAIN(comma) EVAL(comma) R F lead tag"""
import sys; sys.path.insert(0, '/data/Jarrel/expert-predict/src'); from feats import *
train, evl = sys.argv[1].split(','), sys.argv[2].split(','); R = int(sys.argv[3]); Fs = [int(x) for x in sys.argv[4].split(',')]
LEAD = int(sys.argv[5]); tag = sys.argv[6]
SUBS = {'emamix': ([0, 1, 2, 3, 4, 11], False), 'emamix_prior': ([0, 1, 2, 3, 4, 9, 11], False),
        'emamix_stateW': ([0, 1, 2, 3, 4, 11], True), 'full_stateW': (list(range(12)), True),
        'full': (list(range(12)), False)}
pr = priors(train); NF_ = 12
A = {F: np.zeros((2, NL, NF_, NF_)) for F in Fs}; B = {F: np.zeros((2, NL, NF_)) for F in Fs}
lb = -(-LEAD // G)
for t in train:
    P = prep(load(t)); nb = P['C'].shape[0]; nref = -(-nb // (R // G)) + 1
    tg = {F: fut_target(P['C'], R, F, nref, lb) for F in Fs}
    for r, s, X in iter_feats(P, R, pr):
        for F in Fs:
            cs, i0, i1 = tg[F]
            if i1[r] - i0[r] < F // G: continue
            y = (cs[i1[r]] - cs[i0[r]]) / F
            A[F][s] += np.einsum('lef,leg->lfg', X, X); B[F][s] += np.einsum('lef,le->lf', X, y)
    print('trained on', t, flush=True)
Wt = {}
for F in Fs:
    for name, (idx, sw) in SUBS.items():
        w = np.zeros((2, NL, NF_))
        for l in range(NL):
            for s in range(2):
                a = A[F][s, l] if sw else A[F][:, l].sum(0); b = B[F][s, l] if sw else B[F][:, l].sum(0)
                a = a[np.ix_(idx, idx)]; b = b[idx]
                w[s, l, idx] = np.linalg.solve(a + 1e-6 * np.trace(a) / len(idx) * np.eye(len(idx)), b)
        Wt[(name, F)] = w
np.save(f'{W}/results/linw_{tag}.npy', {str(k): v for k, v in Wt.items()}, allow_pickle=True)
rows = []
for t in evl:
    d = load(t); P = prep(d); ex = d['ex']; nb = P['C'].shape[0]; nref = -(-nb // (R // G)) + 1
    S = {k: np.zeros((nref, NL, NE), np.float32) for k in Wt}; S['ema512'] = np.zeros((nref, NL, NE), np.float32)
    for r, s, X in iter_feats(P, R, pr):
        for k, w in Wt.items(): S[k][r] = np.einsum('lef,lf->le', X, w[s])
        S['ema512'][r] = X[..., 2]
    for k, v in S.items():
        for lz in (0, 1):
            o = sim(ex, v, R=R, lead=LEAD, lazy=lz)
            row = dict(task=t, model=str(k), lazy=lz, R=R, lead=LEAD, share=round(o['share'], 4), gbps=round(o['gbps'], 2), N=o['N'])
            rows.append(row); print(json.dumps(row), flush=True)
    del S
with open(f'{W}/results/fit_linear_{tag}.jsonl', 'w') as f:
    for r in rows: f.write(json.dumps(r) + '\n')
