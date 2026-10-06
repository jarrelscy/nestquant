# per-layer oracle coverage curve: share of decode routed slots (non-fixed) captured by the top-k floating experts of the
# true counts of each 64-token window (oracle, upper bound of any predictor); greedy equal-weight allocation of 3825 floating.
import sys,os,json,numpy as np
sys.path.insert(0,'/data/Jarrel/nq-lalloc-wt/tests');sys.path.insert(0,'/data/Jarrel/nq-lalloc-wt/streaming')
os.environ['DECODE']='1'
from test_hostloop_parity import routing_steps
fj=json.load(open('/data/Jarrel/nestquant/threads/22-boundary-experts/fixed_set.json'))
L=list(range(3,78));fx=np.zeros((75,256),bool)
for i,l in enumerate(L):fx[i,fj['fixed_set'][str(l)]]=True
st=routing_steps(int(sys.argv[1]) if len(sys.argv)>1 else 200)
W=64;cov=np.zeros((75,231));tot=np.zeros(75);fxs=np.zeros(75);nw=0;n=np.array([s[1] for s in st])
w=np.zeros((75,256));k=0
def flush(w):
    global tot,fxs,cov,nw
    tot+=w.sum(1);fxs+=(w*fx).sum(1)
    v=np.sort(np.where(fx,-1,w),1)[:,::-1][:,:230];cov[:,1:]+=np.cumsum(np.maximum(v,0),1);nw+=1
for c,nt,_,nr in st:
    if nr and k:flush(w);w[:]=0;k=0
    w+=c;k+=nt
    if k>=W:flush(w);w[:]=0;k=0
cov/=tot[:,None];fxs/=tot
nf=np.zeros(75,int);g=np.diff(cov,axis=1)
import heapq
h=[(-g[i,0],i) for i in range(75)];heapq.heapify(h)
for _ in range(3825):
    x,i=heapq.heappop(h);nf[i]+=1
    if nf[i]<230:heapq.heappush(h,(-g[i,nf[i]],i))
o=dict(windows=nw,tokens=int(n.sum()),fixed_share=fxs.tolist(),cov51=cov[:,51].tolist(),nf_greedy=nf.tolist(),
       hot_u51=(fxs+cov[:,51]).mean(),hot_greedy=float(np.mean([fxs[i]+cov[i,nf[i]] for i in range(75)])))
for k,a in [('R1','R1'),('R2','R2'),('S','S'),('R1r','R1r'),('R2r','R2r'),('Sr','Sr')]:
    a=json.load(open(f'/data/Jarrel/nq-lalloc-wt/streaming/results/nf_layers/{k}.json'))['nf'];o['hot_'+k]=float(np.mean([fxs[i]+cov[i,a[i]] for i in range(75)]))
json.dump(o,open('/data/Jarrel/nq-lalloc/ana/oracle_cov.json','w'))
print('windows',nw,'tokens',int(n.sum()));print('fixed share by band',[round(fxs[a:a+15].mean(),3) for a in range(0,75,15)])
print('oracle hot@51 by band',[round((fxs+cov[:,51])[a:a+15].mean(),3) for a in range(0,75,15)])
print('greedy nf',nf.tolist());print({k:round(v,4) for k,v in o.items() if k.startswith('hot_')})
