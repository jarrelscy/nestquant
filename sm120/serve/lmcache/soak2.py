"""Gate 6: tb4-c2-style soak. Two agent-style chat sessions (own threads) each open with a ~<init>-token context
(together > the ~1.1M-token GPU KV, so they evict each other's prefix blocks), then loop: LLM turn -> 'tool time'
sleep -> append assistant reply + a fresh ~<tool>-token tool output -> next turn. max_num_seqs=1, so turns queue.
Per turn: prompt tokens, GPU prefix hits, LMCache (external) hits, recomputed tokens, latency.
Usage: soak2.py <init_tokens> <tool_tokens> <minutes_after_build>"""
import json, random, sys, threading, time
from lmc_common import *
INIT, TOOL, MIN = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
lock = threading.Lock(); rows = []; errs = []; stop_at = [None]
def chat(msgs):
    t = time.time()
    r = post('/v1/chat/completions', dict(model='local', messages=msgs, max_tokens=256, temperature=0, chat_template_kwargs=dict(enable_thinking=False)))
    return r, time.time() - t
def session(name, delay):
    time.sleep(delay); rnd = random.Random(name)
    code = f"{name}-VAULT-{rnd.randint(1000, 9999)}"
    msgs = [dict(role='system', content=f'You are agent {name}, working in a terminal on a large repository.'),
            dict(role='user', content=f"Repository dump for task {name}:\n" + filler(INIT, name, code, 0.3) + "\n\nFirst: state the secret vault passphrase from the dump, then say what you will run next.")]
    turn = 0
    while True:
        with lock:
            m0 = metrics()
        try:
            r, dt = chat(msgs)
        except Exception as e:
            errs.append(f"{name} turn {turn}: {e!r}"); print('ERROR', errs[-1], flush=True); return
        # metrics deltas are only attributable when the other session is not mid-request; record both views
        m1 = metrics(); txt = r['choices'][0]['message'].get('content') or ''
        u = r['usage']; row = dict(s=name, turn=turn, t=round(time.time() - T0), prompt=u['prompt_tokens'], secs=round(dt, 1),
                                   gpu_hit=m1['prefix_cache_hits_total'] - m0['prefix_cache_hits_total'],
                                   ext_hit=m1['external_prefix_cache_hits_total'] - m0['external_prefix_cache_hits_total'],
                                   needle=code in txt, text=txt[:300])
        with lock: rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "text"}), flush=True)
        if stop_at[0] and time.time() > stop_at[0]: return
        msgs.append(dict(role='assistant', content=txt))
        msgs.append(dict(role='user', content=f"$ tool output (turn {turn}):\n" + filler(TOOL, f'{name}-t{turn}') + "\nContinue: summarise the last output in one line and state the passphrase again."))
        turn += 1; time.sleep(rnd.uniform(5, 25))   # tool execution time
T0 = time.time(); since = time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime())
th = [threading.Thread(target=session, args=('ALPHA', 0)), threading.Thread(target=session, args=('BRAVO', 5))]
for x in th: x.start()
# build ends once both first turns are done; then soak MIN minutes
while sum(1 for r in rows if r['turn'] == 0) < 2 and not errs: time.sleep(5)
stop_at[0] = time.time() + MIN * 60; print(f'build done at {time.time()-T0:.0f}s; soaking {MIN} min', flush=True)
for x in th: x.join()
import re
lgd = {}
for l in loglines(since, r'Reqid: .*Total tokens'):
    m = re.search(r'Reqid: (\S+), Total tokens (\d+), Inference Engine computed tokens: (\d+), LMCache hit tokens: (\d+), need to load: (\d+)', l)
    if m: lgd[m.group(1)] = tuple(map(int, m.groups()[1:]))
lg = list(lgd.values())
tot = sum(a for a, b, c, d in lg); gpu = sum(b for a, b, c, d in lg); ld = sum(d for a, b, c, d in lg)
per_req = dict(requests=len(lg), total_tokens=tot, gpu_prefix_tokens=gpu, lmcache_loaded_tokens=ld, recomputed_tokens=tot - gpu - ld,
               requests_with_lmcache_load=sum(1 for a, b, c, d in lg if d > 256), lmcache_share_of_non_gpu=round(ld / max(1, tot - gpu), 3))
print('per-request (scheduler log):', json.dumps(per_req), flush=True)
json.dump(dict(rows=rows, errs=errs, per_req=per_req, log=lg), open('/data/Jarrel/nq-serve/lmc/soak2.json', 'w'), indent=1)
later = [r for r in rows if r['turn'] > 0]
miss = sum(r['prompt'] - r['gpu_hit'] for r in later); ext = sum(r['ext_hit'] for r in later)
print(json.dumps(dict(turns=len(rows), errors=errs, later_turns=len(later), tokens_not_on_gpu=miss, restored_from_lmcache=ext,
                      lmcache_share_of_gpu_misses=round(ext / miss, 3) if miss else None, max_secs=max(r['secs'] for r in later) if later else None,
                      needles_ok=sum(1 for r in rows if r['needle']), needles_of=len(rows), needle_miss=[(r['s'], r['turn']) for r in rows if not r['needle']])))
