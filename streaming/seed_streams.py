"""(1) Start of request: first 1024 decode tokens of each tb4 request, floating set seeded from the routing of the
prompt's last computed prefill tokens (up to 1024; short or missing prefill fills the remaining places from the default)
vs the default only. Default = floating_default[L] from fixed_set.json if present, else top 51 non-fixed by n_routed.
After the seed, both refresh every 64 tokens from the last 1024 tokens (prefill + decode for the seeded run, decode only
for the default run), 1-window lag. Prefix-cached prompt tokens are not computed, so they are not in the logs.
(2) Several streams: c decode streams from different requests run step-aligned; one shared pool ranked on counts summed
over streams over the last 1024 steps, refresh every 64 steps, 1-window lag. Reports mean route share, the lowest
per-stream share, and SSD GB/s at the aggregate tok/s of the estimate."""
import numpy as np,glob,collections,sys,json
sys.path.insert(0,'/data/Jarrel/nestquant/streaming');import fixed_set
D='/data/Jarrel/routing_logs/glm5.3-arvq-v2';NE=256;L0=3;NL=75;NF=51;WIN=1024;REF=64;MB=9.97e6
fx,_,_=fixed_set.load();fixed=np.zeros((NL,NE),bool)
for l in range(NL):fixed[l,fx[l+L0]]=True
fj=json.load(open(fixed_set.T22))
if 'floating_default' in fj:dflt=[list(fj['floating_default'][str(l+L0)])[:NF] for l in range(NL)];dsrc='floating_default'
else:dflt=[[int(x) for x in np.argsort(-np.where(fixed[l],-1,np.array(fj['n_routed'][str(l+L0)])))[:NF]] for l in range(NL)];dsrc='n_routed top51 non-fixed'
cap=int(sys.argv[1]) if len(sys.argv)>1 else 1_500_000   # decode rows (as decode_churn)
P=collections.defaultdict(list);Dc=collections.defaultdict(list);tot=0
for f in sorted(glob.glob(D+'/seg-*.npz')):
    z=np.load(f);key=z['step'].astype(np.int64)*4096+z['req'];u,inv,cnt=np.unique(key,return_inverse=True,return_counts=True)
    dec=cnt[inv]<=16;r=z['req'];ex=z['experts'][:,L0:L0+NL];pos=z['positions'];ids=z['req_ids'];tag=f.split('-')[2]
    for q in np.unique(r):
        k=r==q;name=str(ids[q])+tag
        if (k&~dec).any():P[name].append((pos[k&~dec],ex[k&~dec]))
        if (k&dec).any():Dc[name].append((pos[k&dec],ex[k&dec]))
    tot+=int(dec.sum())
    if tot>=cap:break
def order(v):
    p=np.concatenate([a for a,_ in v]);e=np.concatenate([b for _,b in v]);o=np.argsort(p,kind='stable');p,e=p[o],e[o]
    k=np.r_[p[1:]!=p[:-1],True];return p[k],e[k]
def onehot(e):
    oh=np.zeros((len(e),NL,NE),np.int32);np.add.at(oh,(np.arange(len(e))[:,None,None],np.arange(NL)[None,:,None],e),1);return oh
def pick(c,extra=None,nfrom=NF):
    """top nfrom non-fixed by counts c, rest from the default list"""
    fl=np.zeros((NL,NE),bool);c=np.where(fixed,-1,c)
    for l in range(NL):
        top=[int(x) for x in np.argsort(-c[l],kind='stable')[:nfrom] if c[l][x]>0]
        for x in dflt[l]:
            if len(top)>=NF:break
            if x not in top and not fixed[l,x]:top.append(x)
        fl[l,top[:NF]]=True
    return fl
# ---- (1) seeding
rows=[];nopre=0
for name,v in Dc.items():
    pd,ed=order(v)
    if len(ed)<WIN:continue
    ed=ed[:WIN];ohd=onehot(ed)
    if name in P:
        pp,ep=order(P[name]);ep=ep[pp<pd[0]][-WIN:]
    else:ep=np.zeros((0,NL,8),np.uint8)
    npre=len(ep);nopre+=npre==0
    ohp=onehot(ep) if npre else np.zeros((0,NL,NE),np.int32)
    out={}
    for mode in ('seed','default'):
        hist=np.concatenate([ohp,ohd]) if mode=='seed' else ohd
        off=npre if mode=='seed' else 0
        cs=np.concatenate([np.zeros((1,NL,NE),np.int64),np.cumsum(hist,0)])
        if mode=='seed':fl0=pick(cs[npre]-cs[max(0,npre-WIN)],nfrom=int(round(NF*min(npre,WIN)/WIN)))
        else:fl0=pick(np.zeros((NL,NE)),nfrom=0)
        sets=[fl0,fl0]                                  # windows 0 and 1 served by the seed (lag 1)
        for s in range(REF,WIN-REF+1,REF):              # set picked at decode token s serves [s+64, s+128)
            a=off+s;c=cs[a]-cs[max(0,a-WIN)]
            sets.append(pick(c,nfrom=NF) if mode=='seed' or s>=REF else fl0)
        sh=[]
        for j in range(WIN//REF):
            hot=sets[j]|fixed;w=ohd[j*REF:(j+1)*REF].sum(0);sh.append((w*hot).sum()/w.sum())
        out[mode]=sh
    rows.append(dict(req=name,npre=npre,seed=out['seed'],default=out['default']))
S=np.array([r['seed'] for r in rows]);Df=np.array([r['default'] for r in rows]);npre=np.array([r['npre'] for r in rows])
r1=dict(default_source=dsrc,requests=len(rows),requests_without_prefill_rows=int(nopre),prefill_rows_median=float(np.median(npre)),
        prefill_ge_1024=int((npre>=WIN).sum()),seed_share_first1024=float(S.mean()),default_share_first1024=float(Df.mean()),
        seed_by_window=[round(float(x),4) for x in S.mean(0)],default_by_window=[round(float(x),4) for x in Df.mean(0)],
        seed_better_frac=float((S.mean(1)>Df.mean(1)).mean()))
print(json.dumps({k:v for k,v in r1.items()},indent=0),flush=True)
# ---- (2) several streams
seqs=[order(v)[1] for v in Dc.values()];seqs=[s for s in seqs if len(s)>=2*WIN+4*REF];rng=np.random.default_rng(0)
TPS={1:111,2:148,4:185,8:214};r2={}
for c in (1,2,4,8):
    ms=[];lo=[];new=[];
    for g in range(min(24,len(seqs)//c)):
        grp=[seqs[i] for i in rng.choice(len(seqs),c,replace=False)];T=min(len(s) for s in grp);T=min(T,6000)
        ohs=[onehot(s[:T]) for s in grp];tot_=sum(ohs);cs=np.concatenate([np.zeros((1,NL,NE),np.int64),np.cumsum(tot_,0)])
        per=[[] for _ in range(c)];prev=None
        for s in range(WIN,T-2*REF+1,REF):
            fl=pick(cs[s]-cs[s-WIN]);hot=fl|fixed
            if prev is not None:new.append(int((fl&~prev).sum()))
            prev=fl
            for i in range(c):w=ohs[i][s+REF:s+2*REF].sum(0);per[i].append((w*hot).sum()/w.sum())
        m=[np.mean(p) for p in per];ms.append(np.mean(m));lo.append(min(m))
    g=np.mean(new)*MB/1e9
    r2[c]=dict(groups=len(ms),mean_share=round(float(np.mean(ms)),4),lowest_stream_mean=round(float(np.mean(lo)),4),
               lowest_stream_min=round(float(np.min(lo)),4),GBps=round(g*TPS[c]/c/REF,2),agg_tps=TPS[c])
    print(c,r2[c],flush=True)
json.dump(dict(seeding=r1,streams=r2),open('/data/Jarrel/nestquant/streaming/results/seed_streams.json','w'),indent=1)
