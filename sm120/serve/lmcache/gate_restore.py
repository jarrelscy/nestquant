"""Gate 1 + 3: cold prefill vs LMCache restore (same boot), temp 0.
  cold    : first sight of the prompt (GPU + LMCache miss, LMCache stores)
  warm    : immediate resend (GPU prefix-cache hit; original KV bytes, control)
  restore : /reset_prefix_cache (GPU only; LMCacheConnectorV1 has no reset_cache, so its CPU tier survives) -> resend
Lossless = restore output == warm output (both consume the SAME stored KV bytes; any diff = LMCache transfer error);
cold-vs-restore/warm differs only by NQ's time-varying decode expert levels. Reports top-20 KLD on first token + common prefix.
Usage: gate_restore.py <tokens> <salt> [out.json|-] [needle_depth]"""
import json, sys, time
from lmc_common import *
N, salt = int(sys.argv[1]), sys.argv[2]
out = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != "-" else f'/data/Jarrel/nq-serve/lmc/restore_{salt}.json'
code = f"ORCHID-{sum(map(ord, salt)) % 9000 + 1000}-KESTREL"
p = "Audit log follows.\n\n" + filler(N, salt, code, float(sys.argv[4]) if len(sys.argv) > 4 else 0.37) + "\n\nQuestion: what is the secret vault passphrase? Answer: The passphrase is"
R = {}
for ph in ['cold', 'warm', 'reset', 'restore', 'warm2']:
    if ph == 'reset': reset_gpu_prefix(); time.sleep(2); continue
    m0 = metrics(); t0 = time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime())
    r = completion(p, max_tokens=32); m1 = metrics()
    r['gpu_hit_tokens'] = m1['prefix_cache_hits_total'] - m0['prefix_cache_hits_total']
    r['ext_hit_tokens'] = m1['external_prefix_cache_hits_total'] - m0['external_prefix_cache_hits_total']
    r['lmc_log'] = [l[-220:] for l in loglines(t0, r'Retrieved|hit tokens|LMCache hit') ][-4:]
    R[ph] = r
    print(f"{ph:8s} {r['prompt_tokens']} tok {r['secs']:8.1f}s gpu_hit {r['gpu_hit_tokens']:.0f} ext_hit {r['ext_hit_tokens']:.0f} needle={'ok' if code in r['text'] else 'MISS'} text={r['text'][:60]!r}", flush=True)
res = dict(tokens=R['cold']['prompt_tokens'], code=code,
           cold_s=round(R['cold']['secs'], 1), restore_s=round(R['restore']['secs'], 1), warm_s=round(R['warm']['secs'], 1),
           speedup=round(R['cold']['secs'] / R['restore']['secs'], 1),
           restore_vs_warm=cmp(R['warm'], R['restore']), restore_vs_warm2=cmp(R['warm2'], R['restore']),
           warm_vs_warm2=cmp(R['warm'], R['warm2']), cold_vs_restore=cmp(R['cold'], R['restore']), cold_vs_warm=cmp(R['cold'], R['warm']),
           needle={k: code in v['text'] for k, v in R.items()}, runs=R)
json.dump(res, open(out, 'w'), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != 'runs'}, indent=1))
