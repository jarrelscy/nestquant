"""Prefill routing windows from the GLM-5.3 routing logs (tb4 agent traffic): for W-token windows of one request's
prefill, per MoE layer: distinct experts touched, and the level-4 pick share when the resident pool (77 = 30%) is
the top-77 of the previous window (first window: static usage top-77) and the SSD budget upgrades the top-k
non-resident experts by in-window count, k = f * 179."""
import numpy as np,glob,json,sys,collections
D='/data/Jarrel/routing_logs/glm5.3-arvq-v2';NE=256;NP=77;L0=3
cap=int(sys.argv[1]) if len(sys.argv)>1 else 3_000_000
reqs=collections.defaultdict(list);tot=0
for f in sorted(glob.glob(D+'/seg-*.npz')):
    z=np.load(f);st=z['step'];rq=z['req']
    key=st.astype(np.int64)*4096+rq;u,inv,cnt=np.unique(key,return_inverse=True,return_counts=True)
    m=cnt[inv]>16
    if not m.any():continue
    ids=z['req_ids'];ex=z['experts'][m][:,L0:];pos=z['positions'][m];r=rq[m]
    for q in np.unique(r):
        k=r==q;reqs[str(ids[q])+f.split('-')[2]].append((pos[k],ex[k]))   # pid in key: req ids restart per server
    tot+=int(m.sum())
    if tot>=cap:break
print('prefill tokens loaded',tot,'requests',len(reqs),flush=True)
seqs=[]
for k,v in reqs.items():
    p=np.concatenate([a for a,_ in v]);e=np.concatenate([b for _,b in v]);o=np.argsort(p,kind='stable')
    p,e=p[o],e[o];keep=np.r_[True,p[1:]!=p[:-1]];seqs.append(e[keep])
nl=seqs[0].shape[1]
glob_cnt=np.zeros((nl,NE),np.int64)
for e in seqs:
    for l in range(nl):glob_cnt[l]+=np.bincount(e[:,l].ravel(),minlength=NE)
static=np.argsort(-glob_cnt,1)[:,:NP]
fs=[0,0.25,0.5,0.75,1.0];res={}
for W in (1024,4096,8192,16384,32768):
    acc=collections.defaultdict(list);nwin=0
    for e in seqs:
        prev=None
        for s in range(0,len(e)-W+1,W):
            w=e[s:s+W];nwin+=1
            for l in range(nl):
                c=np.bincount(w[:,l].ravel(),minlength=NE)
                for pn,pool in (('static',static[l]),('prev',static[l] if prev is None else prev[l])):
                    inp=np.zeros(NE,bool);inp[pool]=True
                    rest=np.sort(c[~inp])[::-1];base=c[inp].sum()
                    for f in fs:
                        k=int(round(f*(NE-NP)));acc[(pn,f)].append((base+rest[:k].sum())/c.sum())
                acc['touched'].append((c>0).mean());acc['touched_nonres'].append(((c>0)&~inp).sum()/(NE-NP))
                acc['max_tok_per_expert'].append(c.max()/(W*8/NE))
            prev=np.stack([np.argsort(-np.bincount(w[:,l].ravel(),minlength=NE))[:NP] for l in range(nl)])
    if not nwin:continue
    r={'windows':nwin,'touched':round(float(np.mean(acc['touched'])),3),'touched_nonres':round(float(np.mean(acc['touched_nonres'])),3),
       'max_over_mean_tokens':round(float(np.mean(acc['max_tok_per_expert'])),2)}
    for pn in ('static','prev'):
        for f in fs:r[f'share_{pn}_f{f}']=round(float(np.mean(acc[(pn,f)])),3)
    res[W]=r;print(W,r,flush=True)
json.dump(res,open('/data/Jarrel/nestquant/streaming/results/prefill_share.json','w'),indent=1)
