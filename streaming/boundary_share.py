"""4-bit share on tb4 decode tokens by distance to the next boundary, with the final fixed set (threads/22 fixed_set.json)
+ 51 floating (top by count over the last 1024 tokens, refresh 64, 1-window lag). Distance d = tokens before the boundary:
</think> is found in the logged input tokens; end of turn (<|user|>/<|observation|> are stop tokens, never fed back in)
is taken as the last logged decode row of each request. Reports fixed-only and fixed+floating route share."""
import numpy as np,glob,collections,sys,json
sys.path.insert(0,'/data/Jarrel/nestquant/streaming');import fixed_set
D='/data/Jarrel/routing_logs/glm5.3-arvq-v2';NE=256;L0=3;NL=75;NF=51;WIN=1024;REF=64;THINK=154842
fx,src,_=fixed_set.load();fixed=np.zeros((NL,NE),bool)
for l in range(NL):fixed[l,fx[l+L0]]=True
reqs=collections.defaultdict(list);tot=0
for f in sorted(glob.glob(D+'/seg-*.npz')):
    z=np.load(f);st=z['step'];rq=z['req'];key=st.astype(np.int64)*4096+rq;u,inv,cnt=np.unique(key,return_inverse=True,return_counts=True)
    m=cnt[inv]<=16
    if not m.any():continue
    ids=z['req_ids'];ex=z['experts'][m][:,L0:L0+NL];pos=z['positions'][m];r=rq[m];tk=z['token_ids'][m]
    for q in np.unique(r):k=r==q;reqs[str(ids[q])+f.split('-')[2]].append((pos[k],ex[k],tk[k]))
    tot+=int(m.sum())
    if tot>=3_000_000:break
B={'all':[],'think_d1':[],'think_d2_4':[],'end_d1':[],'end_d2_4':[]}
acc={k:np.zeros(3) for k in B}   # routes, fixed hits, hot hits
nreq=0
for v in reqs.values():
    p=np.concatenate([a for a,_,_ in v]);e=np.concatenate([b for _,b,_ in v]);t=np.concatenate([c for _,_,c in v])
    o=np.argsort(p,kind='stable');p,e,t=p[o],e[o],t[o];keep=np.r_[p[1:]!=p[:-1],True];p,e,t=p[keep],e[keep],t[keep]   # last row per position
    n=len(e)
    if n<WIN+2*REF:continue
    nreq+=1
    oh=np.zeros((n,NL,NE),np.int32);np.add.at(oh,(np.arange(n)[:,None,None],np.arange(NL)[None,:,None],e),1)
    cs=np.concatenate([np.zeros((1,NL,NE),np.int64),np.cumsum(oh,0)])
    hot=np.zeros((n,NL,NE),bool);hot[:]=fixed
    for s in range(WIN,n-REF+1,REF):                  # set from [s-1024,s) serves [s+64, s+128)
        c=np.where(fixed,-1,cs[s]-cs[s-WIN]);fl=np.zeros_like(fixed);np.put_along_axis(fl,np.argsort(-c,1)[:,:NF],True,1)
        hot[s+REF:s+2*REF]|=fl
    valid=np.zeros(n,bool);valid[WIN+REF:]=True
    dthink=np.full(n,10**9)
    nxt=10**9
    for i in range(n-1,-1,-1):                        # tokens until the next </think> input (row i predicts token i+1)
        if i+1<n and t[i+1]==THINK:nxt=i+1
        dthink[i]=nxt-i
    dend=n-np.arange(n)                                # last row = 1 before end of turn
    sel={'all':valid,'think_d1':valid&(dthink==1),'think_d2_4':valid&(dthink>=2)&(dthink<=4),'end_d1':valid&(dend==1),'end_d2_4':valid&(dend>=2)&(dend<=4)}
    ohf=oh*fixed[None];ohh=oh*hot
    for k,m in sel.items():
        if m.any():acc[k]+=[oh[m].sum(),ohf[m].sum(),ohh[m].sum()]
res={'fixed_set_source':src,'requests':nreq}
for k,a in acc.items():res[k]=dict(rows=int(a[0]//(8*NL)),fixed=round(a[1]/max(a[0],1),4),fixed_plus_floating=round(a[2]/max(a[0],1),4))
print(json.dumps(res,indent=1));json.dump(res,open('/data/Jarrel/nestquant/streaming/results/boundary_share.json','w'),indent=1)
