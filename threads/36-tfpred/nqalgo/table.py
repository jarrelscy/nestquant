"""table.py <fc: jf|gbdt> -> markdown table of cur / val / tap arms per env x task (full-length runs only)"""
import json, os, sys
fc = sys.argv[1] if len(sys.argv) > 1 else 'jf'
RD = ['/data/Jarrel/nq-algo/results/runs', '/data/Jarrel/nq-tfpred/nqalgo/runs']
T = ['embedding-drift-monitor', 'sound-change-cascade', 'freight-dispatch-shift']
E = ['prB6.co', 'lat66.m', 'lat200.m']
A = ['bel', f'cur-{fc}', f'val-{fc}-H256-c0.5', f'tap-{fc}e512-c0.5-H256-mla1', f'tap-{fc}e512-c0.5-Ha2-mla1', f'tap-{fc}w-c0.5-H256-mla1',
     f'tap-{fc}e2048-c0.5-H256-mla1', f'srvtap-{fc}-c0.5-H256-mla1', f'tap-{fc}e512-c0.5-H256-mla1-lsema']


def get(cfg, t):
    for d in RD:
        p = f'{d}/{cfg}.{t}.json'
        if os.path.exists(p): return json.load(open(p))


K = [('share_all', 'slot'), ('sal_all', 'sal'), ('think_all', 'think'), ('ans_all', 'ans'), ('sal_think_all', 'salT'),
     ('sal_ans_all', 'salA'), ('backlog_p90', 'bl90')]
for e in E:
    print(f'\n### {e}\n')
    print('| arm | task | ' + ' | '.join(k[1] for k in K) + ' | d slot vs cur |')
    print('|---' * (len(K) + 3) + '|')
    for a in A:
        for t in T:
            r = get(f'{a}@{e}', t)
            if r is None: continue
            c = get(f'cur-{fc}@{e}', t)
            d = '%+.2f' % (100 * (r['share_all'] - c['share_all'])) if c else ''
            print(f'| {a} | {t.split("-")[0]} | ' + ' | '.join(
                ('%.4f' % r[k] if isinstance(r.get(k), float) and k != 'backlog_p90' else str(r.get(k))) for k, _ in K) + f' | {d} |')
