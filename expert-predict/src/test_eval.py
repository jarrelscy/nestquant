"""Held-out test tasks: hyperparameters were chosen on VAL (formal, embedding); none of these configs are fitted to data."""
import sys, json; sys.path.insert(0, '/data/Jarrel/expert-predict/src'); from answer_pred import *
out = open(f'{W}/results/test_eval.jsonl', 'a')
def rec(t, name, R, lead, o, st, extra={}):
    h = o.pop('hits') / 600.
    r = dict(task=t, pred=name, R=R, lead=lead, share=o['share'], think=float(h[st == 0].mean()), answer=float(h[st == 1].mean()),
             gbps=o['gbps'], sal=o['sal_share'], gain=o['gain_share'], ups_tok=o['ups_tok'], N=o['N'], **extra)
    print(json.dumps(r), flush=True); out.write(json.dumps(r) + '\n'); out.flush()
zero = np.zeros((2, NL, NE), np.float32)
for t in sys.argv[1].split(','):
    d = load(t); ex = d['ex']; P = prep(d); st = P['st']; C = P['C']; N = len(ex)
    for R in (16, 32, 64):
        S = scores(P, R, 512, 2048, 0, 0, 0, zero)                     # plain EMA512
        for lead in (0, 13, 32, 64): rec(t, 'ema512', R, lead, sim(ex, S, R=R, lead=lead), st)
        S = scores(P, R, 256, 2048, 0, 0, 0, zero)
        for lead in (0, 13, 32, 64): rec(t, 'ema256_hm0.1', R, lead, sim(ex, S, R=R, lead=lead, hm=0.1), st)
        if R == 16:
            rec(t, 'ema256xREAP_hm0.25', R, 13, sim(ex, S * REAP.astype(np.float32), R=R, hm=0.25), st)
            rec(t, 'ema256xGAIN_hm0.25', R, 13, sim(ex, S * GAIN.astype(np.float32), R=R, hm=0.25), st)
        del S
        S = scores(P, R, 256, 2048, 0.5, 0, 0, zero)                   # + answer-segment memory
        for lead in (0, 13, 32, 64): rec(t, 'ema256_ans0.5_hm0.1', R, lead, sim(ex, S, R=R, lead=lead, hm=0.1), st)
        del S
    nref = -(-N // 64) + 1; cs = np.concatenate([np.zeros((1, NL, NE), np.float32), np.cumsum(C, 0, dtype=np.float32)])
    i0 = np.minimum(np.arange(nref) * 4 + 1, len(C)); i1 = np.minimum(i0 + 16, len(C)); O = cs[i1] - cs[i0]; del cs
    rec(t, 'oracleF256_cap_lazy', 64, 13, sim(ex, O, lazy=1), st)
    rec(t, 'oracleF256_nocap', 64, 13, sim(ex, O, cap_gbps=1e6), st); del O, P, d
