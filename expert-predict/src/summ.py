import json, collections, sys
rows = [json.loads(l) for f in sys.argv[1:] for l in open(f)]
g = collections.defaultdict(dict)
for r in rows: g[(r['pred'], r['R'], r['hm'], r['lazy'], r['cap'])][r['task']] = r
out = []
for k, v in g.items():
    n = sum(x['N'] for x in v.values()); m = lambda f: sum(x[f] * x['N'] for x in v.values()) / n
    out.append((m('share'), k, m('sal'), m('gain'), m('gbps'), len(v), {t[:10]: round(x['share'], 4) for t, x in v.items()}))
out.sort(key=lambda o: -o[0])
for o in out[:int(__import__('os').environ.get('TOP', 20))]: print('%.4f' % o[0], o[1], 'sal %.4f gain %.4f GBps %.2f n%d' % o[2:6], o[6])
