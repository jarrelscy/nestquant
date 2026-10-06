# live decode speed on prod: tok/s, steps/s (spec drafts), emitted/step; dec-block waits controlled by caller
import json,re,sys,time,urllib.request
K=open('/home/jarrelscy/homeassistant/.env').read();K=re.search(r'VLLM_API_KEY=(\S+)',K).group(1)
H={'Authorization':'Bearer '+K,'Content-Type':'application/json'}
def met():
    t=urllib.request.urlopen(urllib.request.Request('http://localhost:8001/metrics',headers=H)).read().decode()
    g=lambda n:sum(float(m) for m in re.findall(r'^%s\{[^}]*\} (\S+)'%n,t,re.M))
    return g('vllm:spec_decode_num_drafts_total'),g('vllm:spec_decode_num_accepted_tokens_total'),g('vllm:generation_tokens_total')
P=[("code","Write a Python implementation of a red-black tree with insert, delete and search, with docstrings."),
   ("prose","Write a long, detailed essay on the history of the Roman Republic, from its founding to the rise of Augustus.")]
N=int(sys.argv[1]) if len(sys.argv)>1 else 768
out=[]
for name,p in P:
  for rep in range(2):
    d0=met();t0=time.time()
    body=json.dumps({"model":"local","messages":[{"role":"user","content":p}],"max_tokens":N,"temperature":0,"stream":True,
                     "chat_template_kwargs":{"enable_thinking":False}}).encode()
    r=urllib.request.urlopen(urllib.request.Request('http://localhost:8001/v1/chat/completions',data=body,headers=H))
    tf=None
    for ln in r:
        if tf is None and ln.startswith(b'data: {'):tf=time.time()
    t1=time.time();d1=met()
    st=d1[0]-d0[0];gen=d1[2]-d0[2];dt=t1-tf
    out.append(dict(name=name,rep=rep,gen=gen,steps=st,emit_per_step=gen/max(st,1),tps=gen/dt,steps_s=st/dt,ms_step=1000*dt/max(st,1)))
    print(json.dumps(out[-1]),flush=True)
json.dump(out,open(sys.argv[2] if len(sys.argv)>2 else '/dev/null','w'))
