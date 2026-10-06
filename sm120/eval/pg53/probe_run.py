# pg53 stage 1: run probe settings on live serve; one code request each, dump per tag
import json,re,sys,time,os,urllib.request
K=open('/home/jarrelscy/homeassistant/.env').read();K=re.search(r'VLLM_API_KEY=(\S+)',K).group(1)
H={'Authorization':'Bearer '+K,'Content-Type':'application/json'}
def met():
    t=urllib.request.urlopen(urllib.request.Request('http://localhost:8001/metrics',headers=H)).read().decode()
    g=lambda n:sum(float(m) for m in re.findall(r'^%s\{[^}]*\} (\S+)'%n,t,re.M))
    return g('vllm:spec_decode_num_drafts_total'),g('vllm:generation_tokens_total')
def req(p,N=512):
    d0=met()
    body=json.dumps({"model":"local","messages":[{"role":"user","content":p}],"max_tokens":N,"temperature":0,"stream":True,
                     "chat_template_kwargs":{"enable_thinking":False}}).encode()
    r=urllib.request.urlopen(urllib.request.Request('http://localhost:8001/v1/chat/completions',data=body,headers=H));tf=None
    for ln in r:
        if tf is None and ln.startswith(b'data: {'):tf=time.time()
    dt=time.time()-tf;d1=met();st=d1[0]-d0[0];gen=d1[1]-d0[1]
    return dict(gen=gen,steps=st,tps=gen/dt,ms_step=1000*dt/max(st,1))
P="Write a Python implementation of a red-black tree with insert, delete and search, with docstrings."
S=[("off","0 0 0 1"),("ack","1 0 0 1"),("b1s8","1 0 1 8"),("a1s8","1 1 1 8"),("b2s8","1 0 2 8"),("b4s8","1 0 4 8"),("b1s1","1 0 1 1"),("b2s1","1 0 2 1"),("off2","0 0 0 1")]
pre=sys.argv[1] if len(sys.argv)>1 else 'r1'
req(P,64)
res={}
for tag,c in S:
    open('/dev/shm/nq_pgprobe','w').write(c+'\n');time.sleep(1.5)
    r=req(P);res[tag]=r;print(tag,c,json.dumps(r),flush=True)
    open('/dev/shm/nq_pgprobe_dump','w').write(f'{pre}_{tag}\n');time.sleep(2)
open('/dev/shm/nq_pgprobe','w').write('0 0 0 1\n')
json.dump(res,open(f'probe_{pre}.json','w'))
