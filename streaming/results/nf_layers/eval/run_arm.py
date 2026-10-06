# nq-lalloc live forced-decode KLD on the registry panel (copy of nq-kld/reg/live/run.py) + rank-0 per-layer decode hot share
# snapshots (NQ_IOSTATS publish: lay_hot / lay_tot = decode routed slots served at level 4 / all, cumulative) per context.
import os,json,time,re,urllib.request,numpy as np,glob
R='/data/Jarrel/nq-kld/reg';P=R+'/root/panel';DBG='/data/Jarrel/nq-serve/dbg/kld';FK='/dev/shm/nq_kld_force';O='/data/Jarrel/nq-lalloc'
KEY=re.search(r'VLLM_API_KEY=(\S+)',open('/home/jarrelscy/homeassistant/.env').read()).group(1)
TAG=os.environ['TAG']
def post(d,t=3600):
    r=urllib.request.Request('http://localhost:8001/v1/completions',data=json.dumps(d).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer '+KEY})
    return json.loads(urllib.request.urlopen(r,timeout=t).read())
def io0():
    fs=sorted(glob.glob('/dev/shm/nq_io_*_r0.json'),key=os.path.getmtime)
    if not fs:return None
    try:d=json.load(open(fs[-1]))
    except Exception:time.sleep(0.5);d=json.load(open(fs[-1]))
    return dict(lay_hot=d.get('lay_hot'),lay_tot=d.get('lay_tot'),sched=d.get('sched'),t_wall=d.get('t_wall'))
f=open(f'{O}/{TAG}.jsonl','a')
for c in range(25):
    tag=f'{TAG}_c{c:02d}'
    if os.path.exists(f'{DBG}/{tag}/decode.pt'):continue
    tk=np.asarray(json.load(open(f'{P}/tokens/context-{c:04d}.json')),np.int64);tp=f'/dev/shm/nq_regtok_{c}.npy';np.save(tp,tk)
    time.sleep(2.5);a=io0()
    open(FK,'w').write(f'{tag} {tp}');t=time.time()
    try:u=post(dict(model='local',prompt=[int(tk[0])],max_tokens=2047,temperature=0,ignore_eos=True))['usage']
    finally:os.remove(FK)
    dt=time.time()-t
    time.sleep(2.5);b=io0()
    post(dict(model='local',prompt=[int(tk[0])],max_tokens=1))   # flush -> decode.pt
    for _ in range(60):
        if os.path.exists(f'{DBG}/{tag}/decode.pt'):break
        time.sleep(1)
    d=dict(ctx=c,tag=tag,s=round(dt,1),tps=round(2047/dt,1),usage=u,dump=os.path.exists(f'{DBG}/{tag}/decode.pt'),t=time.time(),io_before=a,io_after=b)
    f.write(json.dumps(d)+'\n');f.flush();print(json.dumps({k:v for k,v in d.items() if not k.startswith('io_')}),flush=True)
