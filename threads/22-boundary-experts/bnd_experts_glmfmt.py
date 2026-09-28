# Replication on the new GLM-format capture (native <|user|>/<|observation|> ends, real </think> in c2048).
# Pools every shard that has bnd rows for a layer; doc split = segment_id % 2^20 per group; base rate from new stats0 if present else old.
import numpy as np, json, os, hashlib
root='/tmp/nestquant/19-capture-glmfmt'; old='/tmp/nestquant/19-capture'
plan=json.load(open(f'{root}/plan.json'))['shards']
seg={}
def docs_for(s,rows):
    p=plan[s]; c=p['corpus']; ctx=json.load(open(f'{c}/manifest.json'))['context']
    if c not in seg: seg[c]=np.load(f'{c}/segments.npy',mmap_mode='r')
    w,pos=np.divmod(rows,ctx); d=np.asarray(seg[c][p['fit_start']+w,pos])%(1<<20)
    g=os.path.basename(c)
    return np.array([int(hashlib.md5(f'{g}:{x}'.encode()).hexdigest(),16)&1 for x in d])
out={}
for L in range(3,78):
    H=[];K=[];D=[];HF=[]
    for s in sorted(plan):
        f=f'{root}/bnd_rows/s{int(s):02d}/L{L}/rows.npz'
        if not os.path.exists(f): continue
        r=np.load(f); ids=r['ids'].astype(int); hit=np.zeros((len(ids),256),bool); np.put_along_axis(hit,ids,True,1)
        H.append(hit);K.append(r['kind']);D.append(r['d']);HF.append(docs_for(s,r['row']))
    if not H: continue
    hit=np.concatenate(H);kind=np.concatenate(K);d=np.concatenate(D);half=np.concatenate(HF)
    sp=f'{root}/stats0/L{L}/sal.npy'; bsrc='new' if os.path.exists(sp) else 'old'
    sal=np.load(sp if bsrc=='new' else f'{old}/stats0/L{L}/sal.npy'); base=sal[:,0,0]/(sal[:,0,0].sum()/8)
    res=dict(base_src=bsrc)
    for kc,kn in ((1,'think'),(2,'end')):
        for dname,dm in (('d1',d==1),('d2_4',(d>=2)&(d<=4)),('d1_32',d>=1)):
            m=(kind==kc)&dm
            if m.sum()<20: res[f'{kn}/{dname}']=dict(n=int(m.sum()),consistent=[]); continue
            h=hit[m].mean(0); hA=hit[m&(half==0)].mean(0); hB=hit[m&(half==1)].mean(0); b=np.maximum(base,1e-9)
            cons=np.where((hA>=0.3)&(hB>=0.3)&(hA/b>=3)&(hB/b>=3))[0]
            rho=np.corrcoef(np.argsort(np.argsort(hA/b)),np.argsort(np.argsort(hB/b)))[0,1]
            res[f'{kn}/{dname}']=dict(n=int(m.sum()),rho_halves=float(rho),consistent=[int(e) for e in cons],
                top=[dict(e=int(e),hit=round(float(h[e]),3),lift=round(float(h[e]/b[e]),1)) for e in np.argsort(-(h/b)*(h>=0.1))[:8]])
    out[L]=res
    print(L,bsrc,{k:(v['n'],round(v.get('rho_halves',0),2),v['consistent']) for k,v in res.items() if k!='base_src' and k.endswith(('d1','d2_4'))},flush=True)
json.dump(out,open('/tmp/nestquant/bnd-experts/glmfmt_partial.json','w'))
