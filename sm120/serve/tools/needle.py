"""needle.py TOTAL_TOKENS DEPTH_FRAC -> one long-context retrieval check against localhost:8001 (model local, temp 0)."""
import sys,json,random,time,re,urllib.request
KEY=re.search(r'VLLM_API_KEY=(\S+)',open('/home/jarrelscy/homeassistant/.env').read()).group(1)
def post(p,d,t=7200):
    r=urllib.request.Request('http://localhost:8001'+p,json.dumps(d).encode(),{'Content-Type':'application/json','Authorization':'Bearer '+KEY})
    return json.load(urllib.request.urlopen(r,timeout=t))
ntok=lambda s:post('/tokenize',dict(model='local',prompt=s))['count']
TOT,DEP=int(sys.argv[1]),float(sys.argv[2]);rnd=random.Random(7)
W='river station gauge harbor valley north south orchard copper granite meadow signal ledger archive'.split()
sent=lambda:f"Log {rnd.randint(1000,99999)}: the {rnd.choice(W)} {rnd.choice(W)} reading at site {rnd.randint(1,999)} was {rnd.randint(1,9999)} units. "
blk=''.join(sent() for _ in range(2000));tpb=ntok(blk)
NEEDLE="Important note: the secret passphrase is MAGENTA-FALCON-7319. "
Q="\n\nQuestion: what is the secret passphrase stated in the important note above? Answer: the secret passphrase is"
nb=(TOT-200)//tpb;parts=[''.join(sent() for _ in range(2000)) for _ in range(nb)]
k=int(nb*DEP);text=''.join(parts[:k])+NEEDLE+''.join(parts[k:])+Q
n=ntok(text);pos=ntok(''.join(parts[:k]))
print(f'prompt {n} tokens, needle at token ~{pos} ({pos/n:.3f})',flush=True)
t=time.time();o=post('/v1/completions',dict(model='local',prompt=text,max_tokens=16,temperature=0))
a=o['choices'][0]['text'];ok='MAGENTA-FALCON-7319' in a
print(json.dumps(dict(prompt_tokens=o['usage']['prompt_tokens'],needle_pos=pos,answer=a,retrieved=ok,secs=round(time.time()-t,1))),flush=True)
