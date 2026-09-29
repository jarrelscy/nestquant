"""EMA / REAP-weighted EMA x refresh x hysteresis x lazy sweep. usage: sweep_ema.py TASKS(comma) tag"""
import sys, itertools; sys.path.insert(0, '/data/Jarrel/expert-predict/src'); from common import *
from multiprocessing import Pool
tasks = sys.argv[1].split(','); tag = sys.argv[2]
HLS = [64, 128, 256, 512, 1024]; RS = [16, 32, 64]; HMS = [0, 0.1, 0.25, 0.5, 1.0]
FAC = {'cnt': None, 'reap': REAP.astype(np.float32), 'gain': GAIN.astype(np.float32)}
def work(a):
    task, hl, R, fac = a
    d = load(task); ex = d['ex']; C = block_counts(ex); S = ema_scores(C, R, hl)
    if FAC[fac] is not None: S *= FAC[fac]
    rows = []
    for hm, lz in itertools.product(HMS, (0, 1)):
        if fac != 'cnt' and hm not in (0, 0.25): continue
        o = sim(ex, S, R=R, lead=13, lazy=lz, hm=hm)
        rows.append(dict(task=task, pred=f'ema{hl}_{fac}', R=R, hm=hm, lazy=lz, cap=6, share=o['share'], sal=o['sal_share'],
                         gain=o['gain_share'], gbps=o['gbps'], N=o['N']))
    if fac == 'cnt':
        o = sim(ex, S, R=R, lead=13, cap_gbps=1e6)
        rows.append(dict(task=task, pred=f'ema{hl}_{fac}', R=R, hm=0, lazy=0, cap=0, share=o['share'], sal=o['sal_share'],
                         gain=o['gain_share'], gbps=o['gbps'], N=o['N']))
    return rows
jobs = [(t, h, R, f) for t in tasks for h in HLS for R in RS for f in FAC if not (f != 'cnt' and h not in (128, 256, 512))]
with Pool(14) as p:
    with open(f'{W}/results/sweep_ema_{tag}.jsonl', 'w') as f:
        for rows in p.imap_unordered(work, jobs):
            for r in rows: f.write(json.dumps(r) + '\n')
            f.flush()
