"""Gate 2: cross-session contamination. N sessions share a common 1st part (a shared 'system/repo' prefix, so chunks
diverge mid-stream like real agent sessions), then each has a unique salted body with its own passphrase at a unique depth.
store phase: all N fired concurrently (server runs them one at a time, max_num_seqs=1) -> LMCache stores.
/reset_prefix_cache (GPU) -> restore phase: all N fired again concurrently in reversed order -> every prefix restored from LMCache.
Pass: every reply has its own passphrase and no foreign one, in both phases; restore ext-hit tokens ~= sum of prompts.
Usage: gate_contam.py <n> <shared_tokens> <unique_tokens>"""
import concurrent.futures as cf, json, sys, time
from lmc_common import *
n, SH, UN = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
F = ["MANGO", "PAPAYA", "LYCHEE", "DURIAN", "GUAVA", "LOQUAT", "SAPOTE", "JUJUBE"]; W = ["SATIN", "TWEED", "CHIFFON", "DENIM", "VELOUR", "MUSLIN", "TAFFETA", "BROCADE"]
shared = "You are auditing warehouse logs for several independent sites. Shared site handbook:\n" + filler(SH, 'handbook')
def prompt(i):
    code = f"{F[i % 8]}-{W[(i * 3) % 8]}-{1000 + i * 917}"
    body = filler(UN + i * 1024, f'site{i:02d}', code, 0.15 + 0.1 * i)
    return code, shared + f"\n\nSite {i} log:\n" + body + "\n\nQuestion: what is the exact secret vault passphrase in the site log? Answer: The passphrase is"
P = [prompt(i) for i in range(n)]; codes = {c for c, _ in P}
def fire(i):
    c, p = P[i]; r = completion(p, max_tokens=24, logprobs=1); return i, r
def phase(tag, order):
    m0 = metrics(); t = time.time()
    with cf.ThreadPoolExecutor(n) as ex: out = dict(ex.map(fire, order))
    m1 = metrics(); ok = True
    for i in range(n):
        own = P[i][0] in out[i]['text']; foreign = [c for c in codes - {P[i][0]} if c in out[i]['text']]
        ok &= own and not foreign
        print(f"[{tag}-{i}] {'OK' if own and not foreign else 'FAIL'} own={own} foreign={foreign} {out[i]['prompt_tokens']} tok {out[i]['secs']:.1f}s {out[i]['text'][:48]!r}", flush=True)
    s = dict(wall=round(time.time() - t, 1), prompt_tokens=sum(o['prompt_tokens'] for o in out.values()),
             gpu_hit=m1['prefix_cache_hits_total'] - m0['prefix_cache_hits_total'], ext_hit=m1['external_prefix_cache_hits_total'] - m0['external_prefix_cache_hits_total'], ok=ok)
    print(tag, s, flush=True); return out, s
o1, s1 = phase('store', list(range(n)))
reset_gpu_prefix(); time.sleep(2)
o2, s2 = phase('restore', list(reversed(range(n))))
same = sum(o1[i]['text'] == o2[i]['text'] for i in range(n))
res = dict(store=s1, restore=s2, identical_text=f"{same}/{n}", PASS=s1['ok'] and s2['ok'])
json.dump(dict(res=res, store={i: o1[i]['text'] for i in o1}, restore={i: o2[i]['text'] for i in o2}), open('/data/Jarrel/nq-serve/lmc/contam.json', 'w'), indent=1)
print(json.dumps(res))
