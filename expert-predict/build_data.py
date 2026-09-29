"""Build per-task deduped routing streams (convention of routing-log/budget_pertask_norecompute.py).
Output data/<task>.npz: ex uint8 [N,75,8] (MoE layers L3..L77), tok int32, dec bool (row came from a decode step:
<=16 rows for its request in that step), req int32 (request index within task), in computed order, re-prefilled
context dropped (first occurrence of the 8-gram within its request, per task). Probes -> data/probe-<domain>.npz."""
import glob, json, os, re, collections
from datetime import datetime
from multiprocessing import Pool
import numpy as np
D='/data/Jarrel/routing_logs/glm5.3-arvq-v2'; O='/data/Jarrel/expert-predict/data'; os.makedirs(O,exist_ok=True)
JOB="/home/jarrelscy/homeassistant/benchmarks/glm5.3-arvq-v2-tb40-8h-c1-20260923"
L0,NL=3,75
def grams(t,n=8):
    h=np.zeros(len(t),np.uint64);t=t.astype(np.uint64)+np.uint64(1)
    for k in range(n):
        sh=np.r_[np.zeros(k,np.uint64),t[:len(t)-k]] if k else t
        h=h*np.uint64(1000003)+sh
    return h
def first_seen(h):
    _,f=np.unique(h,return_index=True);k=np.zeros(len(h),bool);k[f]=True;return k
def ts(s):return datetime.fromisoformat(s.replace("Z","+00:00")).timestamp()
wins=[]
for r in glob.glob(f"{JOB}/*/result.json")+glob.glob(f"{JOB}-outage-archives/*/*/result.json"):
    d=json.load(open(r))
    if d.get("started_at") and d["started_at"]>"2026-09-26T12":
        wins.append((ts(d["started_at"]),ts(d["finished_at"]) if d.get("finished_at") else 1e12,r.split("/")[-2].split("__")[0]))
for dd in glob.glob(f"{JOB}/*/"):
    if not os.path.exists(dd+"result.json") and os.path.exists(dd+"agent/trajectory.json"):
        t=json.load(open(dd+"agent/trajectory.json"))
        wins.append((ts(t["steps"][0]["timestamp"])-60,1e12,dd.rstrip("/").split("/")[-1].split("__")[0]))
print(wins)
rx=re.compile(r"^cmpl-replay-(.+?__\w+)-(\d+)-\d+-[0-9a-f]+$");px=re.compile(r"^cmpl-probe-(.+)-(\d+)-")
fs=sorted(glob.glob(D+'/seg-*.npz'))
def one(a):
    fi,f=a;z=np.load(f);names=z['req_ids'];req=z['req'];st=z['step']
    k=st.astype(np.int64)*4096+req;_,inv,c=np.unique(k,return_inverse=True,return_counts=True);dec=c[inv]<=16
    ex=None;out=[]
    for i,n in enumerate(names):
        n=str(n);m=rx.match(n);p=px.match(n)
        if m:key=('replay',m.group(1),int(m.group(2)))
        elif p:key=('probe',p.group(1),int(p.group(2)))
        elif n.startswith('chatcmpl-'):
            s=np.nonzero(req==i)[0];t0=z['step_time'][st[s[0]]]
            w=next((w[2] for w in wins if w[0]<=t0<=w[1]),None)
            key=('live',w or 'UNLABELLED',n)
        else:continue
        if ex is None:ex=z['experts'][:,L0:L0+NL]
        s=np.nonzero(req==i)[0]
        out.append((key,fi,st[s],z['positions'][s],z['token_ids'][s],dec[s],ex[s]))
    return out
parts=collections.defaultdict(list)
with Pool(12) as P:
    for o in P.imap(one,enumerate(fs),chunksize=4):
        for key,*v in o:parts[(key[0],key[1])].append((key[2],*v))
print({k:len(v) for k,v in parts.items()},flush=True)
summ={}
def save(name,ex,tok,dec,rq,ncomp):
    np.savez(f'{O}/{name}.npz',ex=ex,tok=tok,dec=dec,req=rq);summ[name]=dict(computed=int(ncomp),kept=int(len(tok)),dec=int(dec.sum()))
    print(name,summ[name],flush=True)
bytask=collections.defaultdict(list)
for (kind,name),ps in sorted(parts.items()):
    if kind=='replay':   # prior-script convention: sort by (turn,pos), keep last, 8-gram dedupe per turn
        turn=np.concatenate([np.full(len(p[2]),p[0]) for p in ps]);pos=np.concatenate([p[3] for p in ps]).astype(np.int64)
        ex=np.concatenate([p[6] for p in ps]);tok=np.concatenate([p[4] for p in ps]);dec=np.concatenate([p[5] for p in ps])
        key=turn.astype(np.int64)*(1<<32)+pos;o=np.lexsort((np.arange(len(key)),key))
        key,ex,tok,dec=key[o],ex[o],tok[o],dec[o];last=np.r_[key[1:]!=key[:-1],True]
        ex,tok,dec,key=ex[last],tok[last],dec[last],key[last];rt=key>>32
        h=np.concatenate([grams(tok[rt==r]) for r in np.unique(rt)]);f=first_seen(h)
        bytask[name.split('__')[0]].append((ex[f],tok[f],dec[f],rt[f].astype(np.int32),len(h)))
    elif kind=='probe':
        ps=sorted(ps,key=lambda p:(p[0],p[1],p[2][0]))
        ex=np.concatenate([p[6] for p in ps]);tok=np.concatenate([p[4] for p in ps]);dec=np.concatenate([p[5] for p in ps])
        rq=np.concatenate([np.full(len(p[2]),p[0],np.int32) for p in ps])
        save('probe-'+name,ex,tok,dec,rq,len(tok))
    else:  # live: computed order (file, step), keep last row per (req,pos), 8-gram dedupe per request, first seen in task
        names=sorted({p[0] for p in ps});rid={n:j for j,n in enumerate(names)}
        ok=np.concatenate([np.full(len(p[2]),p[1],np.int64)*(1<<32)+p[2] for p in ps])
        rq=np.concatenate([np.full(len(p[2]),rid[p[0]],np.int64) for p in ps]);pos=np.concatenate([p[3] for p in ps]).astype(np.int64)
        ex=np.concatenate([p[6] for p in ps]);tok=np.concatenate([p[4] for p in ps]);dec=np.concatenate([p[5] for p in ps])
        o=np.argsort(ok,kind='stable');ex,rq,pos,tok,dec=ex[o],rq[o],pos[o],tok[o],dec[o]
        k=rq*(1<<24)+pos;_,li=np.unique(k[::-1],return_index=True);keep=np.sort(len(k)-1-li)
        ex,rq,tok,dec=ex[keep],rq[keep],tok[keep],dec[keep];h=np.zeros(len(tok),np.uint64)
        for r in np.unique(rq):m=rq==r;h[m]=grams(tok[m])
        f=first_seen(h);bytask[name].append((ex[f],tok[f],dec[f],rq[f].astype(np.int32)+100000,len(h)))
    parts[(kind,name)]=None
for t,v in bytask.items():
    save(t,np.concatenate([a[0] for a in v]),np.concatenate([a[1] for a in v]),np.concatenate([a[2] for a in v]),
         np.concatenate([a[3] for a in v]),sum(a[4] for a in v))
json.dump(summ,open(f'{O}/summary.json','w'),indent=1)
