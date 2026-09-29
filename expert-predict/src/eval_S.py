"""Run the sim grid on saved score arrays feat/S_<tag>_<task>.npy (R=64 refresh). usage: eval_S.py tag TASKS [R]"""
import sys, itertools; sys.path.insert(0, '/data/Jarrel/expert-predict/src'); from common import *
from multiprocessing import Pool
tag = sys.argv[1]; tasks = sys.argv[2].split(','); R = int(sys.argv[3]) if len(sys.argv) > 3 else 64
HMS = [float(x) for x in os.environ.get('HMS', '0,0.1,0.25,0.5').split(',')]
FACS = os.environ.get('FACS', 'cnt').split(',')
def work(a):
    task, hm, lz, fac = a
    ex = load(task)['ex']; S = np.load(f'{W}/feat/S_{tag}_{task}.npy').astype(np.float32).reshape(-1, NL, NE)
    if fac == 'reap': S *= REAP.astype(np.float32)
    o = sim(ex, S, R=R, lead=int(os.environ.get('LEAD', 13)), lazy=lz, hm=hm)
    return dict(task=task, pred=f'{tag}_{fac}', R=R, hm=hm, lazy=lz, cap=6, share=o['share'], sal=o['sal_share'], gain=o['gain_share'],
                gbps=o['gbps'], N=o['N'])
jobs = list(itertools.product(tasks, HMS, (0, 1), FACS))
with Pool(min(len(jobs), 12)) as p, open(f'{W}/results/evalS_{tag}.jsonl', 'a') as f:
    for r in p.imap_unordered(work, jobs): f.write(json.dumps(r) + '\n'); f.flush(); print(json.dumps(r), flush=True)
