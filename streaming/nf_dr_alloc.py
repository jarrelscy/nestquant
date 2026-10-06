# nq-lalloc arm Dr: per-band KLD sensitivity from the measured arms x per-layer live hot-share gain curve, greedy 3825 floating.
# 1. per arm a (vs U) per context c: dKLD[a,c] = sum_b s_b * dmiss[a,c,b], dmiss = -(live hot share of band b, arm a - U), s_b >= 0 (NNLS).
# 2. per layer live hot curve: live_l(nf) = fixed_l + alpha_l * oracle_cov_l(nf), alpha_l fit (LSQ through the measured arms' (nf_l, live_l)).
# 3. greedy: marginal gain of one more floating expert in layer l = s_band(l)/15 * alpha_l * (cov_l[n+1]-cov_l[n]); cap 230; total 3825.
import json,numpy as np,collections,heapq,sys
import itertools
def nnls(X,y):  # exact NNLS by active-set enumeration (5 vars)
    best=(np.inf,np.zeros(X.shape[1]))
    for m in itertools.product([0,1],repeat=X.shape[1]):
        m=np.array(m,bool);s=np.zeros(X.shape[1])
        if m.any():
            s[m]=np.linalg.lstsq(X[:,m],y,rcond=None)[0]
            if (s<0).any():continue
        r=((y-X@s)**2).sum()
        if r<best[0]:best=(r,s)
    return best[1],best[0]
O='/data/Jarrel/nq-lalloc';NF=f'/data/Jarrel/nq-lalloc-wt/streaming/results/nf_layers'
arms=[a for a in ['R1','R1r','R2','R2r','S','Sr'] if True]
S=collections.defaultdict(dict)
for l in open(f'{O}/scores.jsonl'):x=json.loads(l);S[x['tag'][3:]][x['ctx']]=x['kld']
arms=[a for a in arms if len(S.get(a,{}))==25]
def lay(arm):  # per context per layer (hot,tot) deltas
    R=sorted((json.loads(l) for l in open(f'{O}/la_{arm}.jsonl')),key=lambda r:r['ctx'])
    H=np.array([np.array(r['io_after']['lay_hot'])-r['io_before']['lay_hot'] for r in R],float)
    T=np.array([np.array(r['io_after']['lay_tot'])-r['io_before']['lay_tot'] for r in R],float);return H,T
band=lambda H,T:np.stack([H[:,i:i+15].sum(1)/T[:,i:i+15].sum(1) for i in range(0,75,15)],1)
HU,TU=lay('U');bU=band(HU,TU);kU=np.array([S['U'][c] for c in range(25)])
X=[];y=[];pts=collections.defaultdict(list)
nfU=np.full(75,51);LU=HU.sum(0)/TU.sum(0)
for i in range(75):pts[i].append((51,LU[i]))
for a in arms:
    H,T=lay(a);b=band(H,T);k=np.array([S[a][c] for c in range(25)])
    X.append(-(b-bU));y.append(k-kU)
    nf=json.load(open(f'{NF}/{a}.json'))['nf'];L=H.sum(0)/T.sum(0)
    for i in range(75):pts[i].append((nf[i],L[i]))
X=np.concatenate(X);y=np.concatenate(y)
s5,_=nnls(X,y)  # free 5-band fit (diagnostic only: collinear bands -> degenerate alternating zeros)
print('free 5-band NNLS s_b',np.round(s5,4).tolist())
# regularized: s_b = c0 + c1*(b-2) linear in depth, s_b >= 0 at all bands (2-param LSQ, clipped search over the feasible line)
D=np.stack([X.sum(1),X@(np.arange(5)-2.)],1)
def fitlin(X,y):
    D=np.stack([X.sum(1),X@(np.arange(5)-2.)],1);c=np.linalg.lstsq(D,y,rcond=None)[0]
    if (c[0]+c[1]*(np.arange(5)-2)<0).any():  # project onto boundary c0=2|c1| (one end zero)
        best=None
        for sg in (1,-1):
            z=D[:,0]*2+sg*D[:,1];k=max(0.,(z@y)/(z@z));r=((y-k*z)**2).sum()
            if best is None or r<best[0]:best=(r,np.array([2*k,sg*k]))
        c=best[1]
    return c[0]+c[1]*(np.arange(5)-2)
def nnls(X,y):return fitlin(X,y),0
s,res=nnls(X,y)
# bootstrap over contexts for se
rng=np.random.default_rng(0);bs=[]
na=len(arms)
for _ in range(500):
    cc=rng.integers(0,25,25);idx=np.concatenate([cc+25*j for j in range(na)]);bs.append(nnls(X[idx],y[idx])[0])
bs=np.array(bs)
print('arms',arms,'rows',len(y))
print('KLD per unit band miss share s_b (L3-17/18-32/33-47/48-62/63-77):',np.round(s,4).tolist(),'boot se',np.round(bs.std(0),4).tolist())
pred=X@s;print('fit R2 %.3f'%(1-((y-pred)**2).sum()/((y-y.mean())**2).sum()))
for j,a in enumerate(arms):print(f'  {a}: measured dKLD {y[25*j:25*j+25].mean():+.4f} fit {pred[25*j:25*j+25].mean():+.4f}')
z=np.load(f'{O}/ana/oracle_cov_curve.npz');cov,fxs=z['cov'],z['fxs']
al=np.zeros(75);fl=np.zeros(75)
for i in range(75):
    n=np.array([p[0] for p in pts[i]]);v=np.array([p[1] for p in pts[i]])
    # live = f + alpha*cov(nf); f = fixed share * beta... fit both f and alpha by LSQ (2 params, 7 points)
    A=np.stack([np.ones_like(n,dtype=float),cov[i,n]],1);(f,a_),*_=np.linalg.lstsq(A,v,rcond=None);al[i]=max(a_,0);fl[i]=f
print('alpha by band',[round(al[i:i+15].mean(),3) for i in range(0,75,15)],'(live hot gain per oracle hot gain)')
w=np.repeat(s,15)/15*al;g=np.diff(cov,axis=1)*w[:,None]
nf=np.zeros(75,int);h=[(-g[i,0],i) for i in range(75)];heapq.heapify(h)
LO,HI=11,91   # stay inside the tested per-layer envelope (R2/R2r extremes); live-hot model is unvalidated outside it
nf[:]=LO;h=[(-g[i,LO],i) for i in range(75)];heapq.heapify(h)
for _ in range(3825-75*LO):
    x,i=heapq.heappop(h);nf[i]+=1
    if nf[i]<HI:heapq.heappush(h,(-g[i,nf[i]],i))
assert nf.sum()==3825 and nf.max()<=230
print('Dr nf',nf.tolist());print('Dr band means',[round(nf[i:i+15].mean(),1) for i in range(0,75,15)],'min',nf.min(),'max',nf.max())
pl=lambda nfv:np.array([fl[i]+al[i]*cov[i,nfv[i]] for i in range(75)])
for nm,v in [('U',nfU),('Dr',nf)]+[(a,np.array(json.load(open(f'{NF}/{a}.json'))['nf'])) for a in arms+['D']]:
    p=pl(v);bb=np.array([p[i:i+15].mean() for i in range(0,75,15)]);d=-(bb-np.array([pl(nfU)[i:i+15].mean() for i in range(0,75,15)]))@s
    print(f'  model {nm:3s}: live hot {p.mean():.4f} bands {np.round(bb,3).tolist()} predicted dKLD vs U {d:+.4f}')
if len(sys.argv)>1:
    json.dump(dict(arm='Dr',layers=list(range(3,78)),nf=nf.tolist(),total=3825,
                   note='per-band NNLS KLD sensitivity (arms '+','.join(arms)+') x live/oracle alpha x oracle coverage gain, greedy, linear-in-depth sensitivity, clamp [11,91] (tested envelope), cap 230',
                   s_band=s.tolist()),open(f'{NF}/Dr.json','w'))
    print('wrote',f'{NF}/Dr.json')
