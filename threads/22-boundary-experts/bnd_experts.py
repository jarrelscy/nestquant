# Which routed experts fire consistently just before boundary tokens (</think>, end of turn)?
# Old-corpus chunk 0 (T19 capture): bnd rows carry top-8 ids per row; base rates from sal.npy 'all'.
import numpy as np, json, hashlib, sys
root='/tmp/nestquant/19-capture'; W=json.load
wins=[json.loads(l)['segments'] for l,_ in zip(open('/home/coder/git/orbit-duet/runs/glm53_training_15m_v2/windows.jsonl'),range(2048))]
def doc_of(row):
    w,pos=divmod(int(row),512)
    for s in wins[w]:
        if s['window_offset']<=pos<s['window_offset']+s['tokens']: return s['source_id']
    return 'none'
r0=np.load(f'{root}/bnd_rows/s00/L16/rows.npz')
docs=np.array([doc_of(x) for x in r0['row']]); half=np.array([int(hashlib.md5(d.encode()).hexdigest(),16)&1 for d in docs])
out={}
for L in range(3,78):
    r=np.load(f'{root}/bnd_rows/s00/L{L}/rows.npz'); assert (r['row']==r0['row']).all()
    sal=np.load(f'{root}/stats0/L{L}/sal.npy'); base=sal[:,0,0]/(sal[:,0,0].sum()/8)   # per-token hit prob
    ids=r['ids'].astype(int); hit=np.zeros((len(ids),256),bool); np.put_along_axis(hit,ids,True,1)
    res={}
    for kind,kn in ((1,'think'),(2,'end')):
        for dname,dm in (('d1',r['d']==1),('d2_4',(r['d']>=2)&(r['d']<=4)),('d1_32',r['d']>=1)):
            m=(r['kind']==kind)&dm
            h=hit[m].mean(0); hA=hit[m&(half==0)].mean(0); hB=hit[m&(half==1)].mean(0)
            lift=h/np.maximum(base,1e-9); liftA=hA/np.maximum(base,1e-9); liftB=hB/np.maximum(base,1e-9)
            # consistent: fires on >=30% of rows, >=3x its base rate, in BOTH document halves
            cons=np.where((hA>=0.3)&(hB>=0.3)&(liftA>=3)&(liftB>=3))[0]
            rho=np.corrcoef(np.argsort(np.argsort(liftA)),np.argsort(np.argsort(liftB)))[0,1]
            res[f'{kn}/{dname}']=dict(n=int(m.sum()),nA=int((m&(half==0)).sum()),nB=int((m&(half==1)).sum()),rho_halves=float(rho),
                top=[dict(e=int(e),hit=round(float(h[e]),3),base=round(float(base[e]),4),lift=round(float(lift[e]),1),hitA=round(float(hA[e]),3),hitB=round(float(hB[e]),3)) for e in np.argsort(-lift*(h>=0.1))[:8]],
                consistent=[int(e) for e in cons])
    out[L]=res
    t=res['think/d1']; e=res['end/d1']
    print(f"L{L:2d} think d1 n={t['n']:4d} rho={t['rho_halves']:.2f} cons={t['consistent']} top={[(x['e'],x['hit'],x['lift']) for x in t['top'][:3]]} | end d1 n={e['n']} rho={e['rho_halves']:.2f} cons={e['consistent']} top={[(x['e'],x['hit'],x['lift']) for x in e['top'][:3]]}",flush=True)
json.dump(out,open('/tmp/nestquant/bnd-experts/old_chunk0.json','w'))
