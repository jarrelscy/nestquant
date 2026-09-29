import sys; sys.path.insert(0, '/data/Jarrel/expert-predict/src'); from common import *
task = sys.argv[1]; d = load(task); ex = d['ex']; N = len(ex)
C = block_counts(ex); R = 64; nref = -(-N // R) + 1
res = {}
S = ema_scores(C, R, 512); res['ema512'] = sim(ex, S); res['ema512_lazy'] = sim(ex, S, lazy=1)
res['ema512_nocap'] = sim(ex, S, cap_gbps=1e6)
for W in (64, 256, 1024):
    O = window_counts(ex, R, 13, nref) if W == 64 else None
    if O is None:  # future window [t+13, t+13+W)
        cs = np.concatenate([np.zeros((1, NL, NE), np.float32), np.cumsum(C, 0, dtype=np.float32)])
        nb = C.shape[0]; i0 = np.minimum(np.arange(nref) * 4 + 1, nb); i1 = np.minimum(i0 + W // 16, nb)
        O = cs[i1] - cs[i0]
    for lz in (0, 1):
        res[f'oracleF{W}_lag13_cap6_lazy{lz}'] = sim(ex, O, lazy=lz)
    res[f'oracleF{W}_lag13_nocap'] = sim(ex, O, cap_gbps=1e6)
O0 = window_counts(ex, R, 0, nref); res['oracle_lead0_nocap'] = sim(ex, O0, lead=0, cap_gbps=1e6)
out = {k: (round(v['share'], 4), round(v['gbps'], 2)) for k, v in res.items()}
print(task, N, out, flush=True)
json.dump(dict(task=task, N=N, res=out), open(f'{W}/results/baseline_{task}.json', 'w'))
