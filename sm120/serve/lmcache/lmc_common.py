"""Shared helpers for the NQ LMCache gates (API key read from homeassistant .env, never printed)."""
import json, math, re, subprocess, time, urllib.request
K = re.search(r'VLLM_API_KEY=(\S+)', open('/home/jarrelscy/homeassistant/.env').read()).group(1)
H = {'Authorization': 'Bearer ' + K, 'Content-Type': 'application/json'}
B = 'http://localhost:8001'
C = 'glm53-nestquant'

def post(path, body=None, timeout=7200):
    d = None if body is None else json.dumps(body).encode()
    r = urllib.request.urlopen(urllib.request.Request(B + path, headers=H, data=d if d is not None else b'', method='POST'), timeout=timeout)
    t = r.read().decode()
    return json.loads(t) if t.strip().startswith(('{', '[')) else t

def ntok(text):
    return post('/tokenize', dict(model='local', prompt=text))['count']

def reset_gpu_prefix():
    post('/reset_prefix_cache')

def filler(n_tok, salt, needle=None, depth=0.5):
    """~n_tok tokens of distinct log lines (salted), optional needle line at depth."""
    n = max(50, int(n_tok / 34.5))
    L = [f"Line {i:07d} [{salt}]: warehouse {chr(65 + (i * 7 + len(salt)) % 26)}{(i * 31) % 97} logged {100 + (i * 7919 + sum(map(ord, salt)) * 37 % 1000) % 900} crate transfers, "
         f"shift {(i + 3) % 3}, checksum {(i * 2654435761 + sum(map(ord, salt))) % 1000003:07d}." for i in range(n)]
    if needle: L[int(n * depth)] = f"Line {int(n * depth):07d} [{salt}]: IMPORTANT: the secret vault passphrase is {needle}. Remember it."
    return "\n".join(L)

def completion(prompt, max_tokens=24, logprobs=20, **kw):
    t = time.time()
    r = post('/v1/completions', dict(model='local', prompt=prompt, max_tokens=max_tokens, temperature=0, logprobs=logprobs, **kw))
    dt = time.time() - t
    c = r['choices'][0]
    return dict(text=c['text'], secs=dt, prompt_tokens=r['usage']['prompt_tokens'],
                top=[dict(d) for d in c['logprobs']['top_logprobs']], toks=c['logprobs']['tokens'])

def kld_top(p, q):
    """KL(p||q) over p's top-k (renormalised on the union, missing entries floored at q's min)."""
    keys = set(p) | set(q); fp = min(p.values()); fq = min(q.values())
    P = {k: math.exp(p.get(k, fp - 5)) for k in keys}; Q = {k: math.exp(q.get(k, fq - 5)) for k in keys}
    sp, sq = sum(P.values()), sum(Q.values())
    return sum(P[k] / sp * math.log((P[k] / sp) / (Q[k] / sq)) for k in keys)

def cmp(a, b):
    n = min(len(a['toks']), len(b['toks'])); same = 0
    while same < n and a['toks'][same] == b['toks'][same]: same += 1
    k = [kld_top(a['top'][i], b['top'][i]) for i in range(min(same + 1, n))]
    return dict(identical=a['text'] == b['text'], common_prefix_tokens=same, of=n, kld_first=round(k[0], 6), kld_max_on_common=round(max(k), 6))

def metrics():
    t = urllib.request.urlopen(urllib.request.Request(B + '/metrics', headers=H)).read().decode()
    out = {}
    for n in ['prefix_cache_queries_total', 'prefix_cache_hits_total', 'external_prefix_cache_queries_total', 'external_prefix_cache_hits_total', 'prompt_tokens_total']:
        out[n] = sum(float(v) for v in re.findall(rf'^vllm:{n}\{{[^}}]*\}} (\S+)', t, re.M))
    return out

def loglines(since, pat):
    o = subprocess.run(['docker', 'logs', '--since', since, C], capture_output=True, text=True)
    return [l for l in (o.stdout + o.stderr).splitlines() if re.search(pat, l)]
