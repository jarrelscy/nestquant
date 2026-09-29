"""Session-restore gate (nq_session.py): 2 tb4 sessions alternating through the chat API, arm off then arm on (in-boot
toggle /dev/shm/nq_sr_ctl), same turn plan in both arms:
  A1 B1 (first turns, unknown sessions) | A2 B2 A3 B3 (short observations behind a prefix / LMCache hit) |
  A4 (+~4K-token observation) B4 (+~64K-token observation) | A5 B5 (short)
Observations are the tasks' real terminal outputs from a tb4 run; the long ones are real source text. Every observation
starts with an arm nonce, so arm on never reuses arm off's KV beyond the shared prompt prefix.
Per request the server's rows (NQ_SR_OUT = /data/Jarrel/nq-serve/dbg/sr_stats.jsonl) are collected and summarized:
served level-4 share of the first 64 decode tokens after a known-session switch (off vs on), restore landed before
decode at 4K / 64K, arrival -> landed and its overlap with prefill for short / 4K / 64K turns.
usage: gate_sr.py [out.json] [task_a task_b]"""
import json,re,sys,time,glob,os,urllib.request,statistics as ST
K=re.search(r'VLLM_API_KEY=(\S+)',open('/home/jarrelscy/homeassistant/.env').read()).group(1)
H={'Authorization':'Bearer '+K,'Content-Type':'application/json'}
J='/home/jarrelscy/homeassistant/benchmarks/glm5.3-arvq-v2-tb40-8h-c1-20260923'
ROWS='/data/Jarrel/nq-serve/dbg/sr_stats.jsonl';CTL='/dev/shm/nq_sr_ctl'
OUT=sys.argv[1] if len(sys.argv)>1 else '/data/Jarrel/nq-serve/gate_sr.json'
TA,TB=(sys.argv[2],sys.argv[3]) if len(sys.argv)>3 else ('fin-saccr-rwa','formal-crypto')
def task(name):
    tj=glob.glob(f'{J}/{name}*/agent/trajectory.json')[0];st=json.load(open(tj))['steps']
    u1=st[0]['message'];obs=[]
    for s in st[1:]:
        for r in ((s.get('observation') or {}).get('results') or []):
            c=r.get('content')
            if isinstance(c,str) and c.strip():obs.append(c[-1500:])
    return u1,obs
def filler(ntok):
    txt='';fs=sorted(glob.glob('/data/Jarrel/nestquant/**/*.py',recursive=True))
    for f in fs:
        txt+=f'\n$ cat {f}\n'+open(f,errors='ignore').read()
        if len(txt)>ntok*3.2:break
    return txt[:int(ntok*3.2)]
def chat(msgs,n=160):
    t=time.time()
    r=json.load(urllib.request.urlopen(urllib.request.Request('http://localhost:8001/v1/chat/completions',headers=H,data=json.dumps(dict(
        model='local',messages=msgs,max_tokens=n,temperature=0)).encode()),timeout=1800))
    m=r['choices'][0]['message'];return (m.get('content') or '')+'',r['usage'],time.time()-t
def arm(on,nonce):
    open(CTL,'w').write(f'on={int(on)} reset=1\n');time.sleep(0.2)
    S={nm:dict(u1=u1,obs=obs,msgs=[{'role':'user','content':u1}],i=0) for nm,(u1,obs) in (('A',task(TA)),('B',task(TB)))}
    plan=[('A',0),('B',0)]+[('A','s'),('B','s')]*2+[('A',4096),('B',65536)]+[('A','s'),('B','s')]
    n0=sum(1 for _ in open(ROWS)) if os.path.exists(ROWS) else 0;info=[]
    for nm,kind in plan:
        s=S[nm]
        if kind!=0:
            o=s['obs'][s['i']%max(1,len(s['obs']))] if s['obs'] else '$ ls\n';s['i']+=1
            if kind!='s':o=o[:400]+'\n'+filler(kind)
            s['msgs'].append({'role':'user','content':f'[run {nonce}] New Terminal Output:\n'+o})
        txt,u,dt=chat(s['msgs'])
        s['msgs'].append({'role':'assistant','content':txt or 'ok'})
        info.append(dict(sess=nm,kind=str(kind),prompt=u['prompt_tokens'],cached=(u.get('prompt_tokens_details') or {}).get('cached_tokens'),
                         completion=u['completion_tokens'],s=round(dt,2)))
        print(json.dumps(info[-1]),flush=True)
    time.sleep(3)
    rows=[json.loads(l) for l in open(ROWS)][n0:]
    return info,rows
def summ(info,rows):
    out=[];rows=list(rows)
    for i in info:
        j=next((k for k,r in enumerate(rows) if r['plen']==i['prompt']),None);r=rows.pop(j) if j is not None else {}
        out.append(dict(i,**{k:r.get(k) for k in ('switch','known','restored','on','ups','downs','cached','new','ms_start','ms_landed',
        'ms_decode_start','landed_before_decode','overlap','win_tok','win_share4','pf_tps','la_ups')}))
    return out
res={}
for nm,on,nonce in (('off',False,'off-%d'%time.time()),('on',True,'on-%d'%time.time())):
    info,rows=arm(on,nonce);res[nm]=summ(info,rows)
os.path.exists(CTL) and os.remove(CTL)
g=lambda arm,f:[r for r in res[arm] if f(r)]
ks=lambda r:r['kind']=='s' and r.get('known')
sh=lambda arm,f:round(ST.mean([r['win_share4'] for r in g(arm,f) if r.get('win_share4') is not None] or [float('nan')]),4)
summary=dict(short_known_first64_share=dict(off=sh('off',ks),on=sh('on',ks)),
             long_known_first64_share={k:dict(off=sh('off',lambda r,k=k:r['kind']==k),on=sh('on',lambda r,k=k:r['kind']==k)) for k in ('4096','65536')},
             restore={k:[{x:r.get(x) for x in ('new','cached','ups','downs','ms_start','ms_landed','ms_decode_start','landed_before_decode','overlap')}
                         for r in g('on',lambda r,k=k:r['kind']==k and r.get('known'))] for k in ('s','4096','65536')})
json.dump(dict(summary=summary,arms=res),open(OUT,'w'),indent=1)
print(json.dumps(summary,indent=1))
