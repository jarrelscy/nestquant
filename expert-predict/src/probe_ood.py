"""OOD check on the 21 probe domains (prefill text streamed token-by-token through the same sim; chunks concatenated per domain)."""
import sys, glob, json, collections; sys.path.insert(0, '/data/Jarrel/expert-predict/src'); from common import *
dom = collections.defaultdict(list)
for f in sorted(glob.glob(f'{DATA}/probe-*.npz')): dom[f.split('probe-')[1].rsplit('-', 1)[0]].append(f)
out = open(f'{W}/results/probe_ood.jsonl', 'w')
for k, fs in dom.items():
    ex = np.ascontiguousarray(np.concatenate([np.load(f)['ex'] for f in fs])); N = len(ex); C = block_counts(ex)
    res = {}
    res['ema512_R64'] = sim(ex, ema_scores(C, 64, 512))
    res['ema256_R16_hm0.1'] = sim(ex, ema_scores(C, 16, 256), R=16, hm=0.1)
    res['ema256xREAP_R16_hm0.25'] = sim(ex, ema_scores(C, 16, 256) * REAP.astype(np.float32), R=16, hm=0.25)
    nref = -(-N // 64) + 1; cs = np.concatenate([np.zeros((1, NL, NE), np.float32), np.cumsum(C, 0, dtype=np.float32)])
    i0 = np.minimum(np.arange(nref) * 4 + 1, len(C)); i1 = np.minimum(i0 + 16, len(C)); O = cs[i1] - cs[i0]
    res['oracleF256_cap_lazy'] = sim(ex, O, lazy=1)
    rec = dict(domain=k, N=N, **{n: dict(share=round(r['share'], 4), sal=round(r['sal_share'], 4), gbps=round(r['gbps'], 2)) for n, r in res.items()})
    print(json.dumps(rec), flush=True); out.write(json.dumps(rec) + '\n')
