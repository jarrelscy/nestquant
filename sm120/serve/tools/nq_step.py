"""single-stream decode: tok/s, accepted tokens/step, ms/step (from vLLM spec-decode counters). Usage: nq_step.py [label] [max_tokens]"""
import json,re,sys,time,urllib.request
K=re.search(r'VLLM_API_KEY=(\S+)',open('/home/jarrelscy/homeassistant/.env').read()).group(1)
H={'Authorization':'Bearer '+K,'Content-Type':'application/json'}
def met():
    t=urllib.request.urlopen(urllib.request.Request('http://localhost:8001/metrics',headers=H)).read().decode()
    g=lambda n:sum(float(v) for v in re.findall(rf'^vllm:{n}\{{[^}}]*\}} (\S+)',t,re.M))
    return g('spec_decode_num_drafts_total'),g('spec_decode_num_accepted_tokens_total')
P=["Implement a thread-safe LRU cache in Python with get/put and TTL expiry, with tests.",
   "Write a detailed essay on the history of the Roman Empire.",
   "Write a C++ program that parses a CSV file, computes per-column statistics and prints a table.",
   "Explain how TCP congestion control works, covering slow start, AIMD, and BBR."]
N=int(sys.argv[2]) if len(sys.argv)>2 else 1024;rows=[]
for p in P:
    d0,a0=met();t=time.time()
    r=json.load(urllib.request.urlopen(urllib.request.Request('http://localhost:8001/v1/chat/completions',headers=H,data=json.dumps(dict(model='local',messages=[{'role':'user','content':p}],max_tokens=N,temperature=0,chat_template_kwargs=dict(enable_thinking=False))).encode())))
    dt=time.time()-t;d1,a1=met();c=r['usage']['completion_tokens'];st=d1-d0
    rows.append((c/dt,c/st if st else 0,dt*1e3/st if st else 0))
    print(f'  {c} tok {c/dt:6.1f} tok/s  {c/max(st,1):.2f} tok/step  {dt*1e3/max(st,1):.2f} ms/step',flush=True)
import statistics as S
print(json.dumps(dict(label=sys.argv[1] if len(sys.argv)>1 else '',tok_s=round(S.mean(r[0] for r in rows),1),tok_step=round(S.mean(r[1] for r in rows),2),ms_step=round(S.median(r[2] for r in rows),2))))
