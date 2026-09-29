"""C1 soak: mixed prompts at concurrency 1 / 2 / 4 (cycling every PHASE s) for MIN minutes against :8001; every
PROBE s a greedy coherence probe (Paris + distinct-8gram of a story). JSON lines -> soak.jsonl, summary on the last line.
  soak.py MIN [PHASE=300] [PROBE=300]"""
import json,os,re,sys,time,threading,urllib.request,random
D=os.path.dirname(os.path.abspath(__file__));MIN=float(sys.argv[1]);PH=float(sys.argv[2]) if len(sys.argv)>2 else 300;PR=float(sys.argv[3]) if len(sys.argv)>3 else 300
K=re.search(r'VLLM_API_KEY=(\S+)',open('/home/jarrelscy/homeassistant/.env').read()).group(1)
H={'Authorization':'Bearer '+K,'Content-Type':'application/json'}
P=['Write a Python function that parses an ISO-8601 duration string and returns seconds. Include tests.',
   'Explain how a B-tree insertion works, step by step, with a small example.',
   'Summarise the causes and consequences of the 1929 stock market crash in about 300 words.',
   'Write a bash script that finds the 10 largest files under a directory, skipping symlinks.',
   'Prove that there are infinitely many primes, then give two different proofs.',
   'Translate into French and explain the grammar: "I would have gone if you had asked me."',
   'Design a SQL schema for a library lending system and write three example queries.',
   'Write a short story about a cartographer who maps a city that keeps changing.',
   'Implement Dijkstra in C++ with a binary heap and explain its complexity.',
   'What are the trade-offs between TCP and QUIC? Be concrete.']
def chat(p,n,temp=0.7,think=True):
    kw={} if think else dict(chat_template_kwargs=dict(enable_thinking=False))   # probes: no reasoning, as coh.py
    t=time.time();r=json.load(urllib.request.urlopen(urllib.request.Request('http://localhost:8001/v1/chat/completions',headers=H,data=json.dumps(dict(
        model='local',messages=[{'role':'user','content':p}],max_tokens=n,temperature=temp,**kw)).encode()),timeout=900))
    return r['choices'][0]['message']['content'] or '',r['usage']['completion_tokens'],time.time()-t
out=open(D+'/soak.jsonl','a');lk=threading.Lock()
def w(d):
    with lk:out.write(json.dumps(d)+'\n');out.flush()
t0=time.time();end=t0+60*MIN;st=dict(req=0,err=0,tok=0,probes=0,probe_fail=0);errs=[]
def worker(i,conc):
    rng=random.Random(i*7919+int(time.time()))
    while time.time()<end and cur[0]==conc:
        try:
            s,n,dt=chat(rng.choice(P),rng.choice([256,512,1024]));st['req']+=1;st['tok']+=n
            w(dict(t=round(time.time()-t0,1),conc=conc,tok=n,s=round(dt,2),tps=round(n/dt,1)))
        except Exception as e:
            st['err']+=1;errs.append(repr(e)[:200]);w(dict(t=round(time.time()-t0,1),conc=conc,error=repr(e)[:200]))
def probe():
    while time.time()<end:
        time.sleep(PR)
        try:
            # (the first version sent these with thinking on: the 300-token story budget went to reasoning, content
            #  came back empty and every probe 'failed' -- a checker bug; the probes now match coh.py)
            a,_,_=chat('What is the capital of France? Answer in one word.',512,0,False);s,_,_=chat('Write a short story (about 250 words) about a lighthouse keeper who finds a message in a bottle.',300,0,False)
            ws=s.split();g=[tuple(ws[i:i+8]) for i in range(max(0,len(ws)-7))];dr=len(set(g))/max(1,len(g));ok='paris' in a.lower() and len(ws)>120 and dr>0.9
            st['probes']+=1;st['probe_fail']+=not ok;w(dict(t=round(time.time()-t0,1),probe=ok,distinct8=round(dr,3),words=len(ws),paris=a.strip()[-60:],story_head=s[:120]))
        except Exception as e:st['probe_fail']+=1;w(dict(t=round(time.time()-t0,1),probe_error=repr(e)[:200]))
cur=[0];threading.Thread(target=probe,daemon=True).start();k=0
while time.time()<end:
    conc=[1,2,4][k%3];k+=1;cur[0]=conc;ts=[threading.Thread(target=worker,args=(j+10*k,conc)) for j in range(conc)]
    [t.start() for t in ts];time.sleep(max(0,min(PH,end-time.time())));cur[0]=-1;[t.join() for t in ts]
w(dict(summary=True,minutes=round((time.time()-t0)/60,1),**st,errors=errs[:5]))
print(json.dumps(dict(minutes=round((time.time()-t0)/60,1),**st)))
