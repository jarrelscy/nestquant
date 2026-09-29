"""Share vs AGGREGATE I/O cap (GB/s, all 4 ranks; = 4x per-rank). Coordinator correction: live cap = 6 GB/s PER RANK = 24 aggregate.
Old runs used 6 aggregate. Predictors fixed (hyperparameters from val at 6 agg; hm/hl variants included for high caps)."""
import sys, json, os; sys.path.insert(0, '/data/Jarrel/expert-predict/src'); from answer_pred import *
CAPS = tuple(float(x) for x in os.environ.get('CAPS', '3,6,12,24,48,1e6').split(','))
import os
done = set()
for l in open(f'{W}/results/cap_curve.jsonl'):
    r = json.loads(l); done.add((r['task'], r['pred'], r['cap'], r['lazy']))
out = open(f'{W}/results/cap_curve.jsonl', 'a'); zero = np.zeros((2, NL, NE), np.float32)
def rec(t, name, R, cap, lazy, o, st):
    h = o.pop('hits') / 600.
    r = dict(task=t, pred=name, R=R, cap=cap, lazy=lazy, share=o['share'], think=float(h[st == 0].mean()), answer=float(h[st == 1].mean()),
             gbps=o['gbps'], sal=o['sal_share'], N=o['N'])
    print(json.dumps(r), flush=True); out.write(json.dumps(r) + '\n'); out.flush()
for t in sys.argv[1].split(','):
    d = load(t); ex = d['ex']; P = prep(d); st = P['st']; C = P['C']; N = len(ex)
    preds = [('ema512', 64, 512, 0, [0.0]), ('ema256', 16, 256, 0, [0.0, 0.1]), ('ema128', 16, 128, 0, [0.0, 0.1]),
             ('ema256_ans0.5', 16, 256, 0.5, [0.0, 0.1]), ('ema128_ans0.5', 16, 128, 0.5, [0.0])]
    for name, R, hl, a, hms in preds:
        if all((t, f'{name}_R{R}_hm{hm}', cap, 0) in done for hm in hms for cap in CAPS): continue
        S = scores(P, R, hl, 2048, a, 0, 0, zero)
        for hm in hms:
            for cap in CAPS:
                if (t, f'{name}_R{R}_hm{hm}', cap, 0) in done: continue
                rec(t, f'{name}_R{R}_hm{hm}', R, cap, 0, sim(ex, S, R=R, hm=hm, cap_gbps=cap), st)
        del S
    nref = -(-N // 64) + 1; cs = np.concatenate([np.zeros((1, NL, NE), np.float32), np.cumsum(C, 0, dtype=np.float32)])
    for F in (64, 256):
        i0 = np.minimum(np.arange(nref) * 4 + 1, len(C)); i1 = np.minimum(i0 + F // 16, len(C)); O = cs[i1] - cs[i0]
        for cap in CAPS:
            for lz in ((0, 1) if cap < 1e5 else (0,)):
                if (t, f'oracleF{F}', cap, lz) in done: continue
                rec(t, f'oracleF{F}', 64, cap, lz, sim(ex, O, lazy=lz, cap_gbps=cap), st)
        del O
    del cs, P, d
