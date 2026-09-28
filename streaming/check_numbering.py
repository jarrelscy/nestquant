"""Expert-numbering check between the thread-22 capture and serving routing logs: per layer Spearman rho of the
capture's n_routed[L] against tb4 usage and against the 19 probe corpora, plus fixed-set route coverage.
rho ~0.3+ = same numbering, ~0 = mismatch."""
import numpy as np,json,glob,sys
F=sys.argv[1] if len(sys.argv)>1 else '/data/Jarrel/nestquant/threads/22-boundary-experts/fixed_set.json'
def rk(x):return np.argsort(np.argsort(x,kind='stable'),kind='stable').astype(float)
def sp(a,b):return float(np.corrcoef(rk(a),rk(b))[0,1])
d=json.load(open(F))
c=np.zeros((78,256),np.int64)
for f in sorted(glob.glob('/data/Jarrel/routing_logs/glm5.3-arvq-v2/seg-*.npz'))[:60]:
    x=np.load(f)['experts']
    for i in range(3,78):c[i]+=np.bincount(x[:,i].ravel(),minlength=256)
pc=np.load('/home/jarrelscy/homeassistant/routing-log/probe_domain_counts.npz')['counts'].sum(0)   # [75,256], layers 3..77
out={}
for L in range(3,78):
    n=np.array(d['n_routed'][str(L)]);fs=d['fixed_set'][str(L)]
    out[L]=dict(rho_tb4=sp(n,c[L]),rho_probes=sp(n,pc[L-3]),rho_tb4_probes=sp(c[L],pc[L-3]),
                cov_tb4=float(c[L,fs].sum()/c[L].sum()),cov_probes=float(pc[L-3,fs].sum()/pc[L-3].sum()))
for L in (3,10,20,40,60,77):print('L%d'%L,{k:round(v,3) for k,v in out[L].items()})
print('median',{k:round(float(np.median([o[k] for o in out.values()])),3) for k in out[3]})
json.dump(out,open('/data/Jarrel/nestquant/streaming/results/check_numbering.json','w'),indent=1)
