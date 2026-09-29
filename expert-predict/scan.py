import numpy as np,glob,collections,json,sys
from multiprocessing import Pool
fs=sorted(glob.glob('/data/Jarrel/routing_logs/glm5.3-arvq-v2/seg-*.npz'))
def one(f):
    z=np.load(f);r=z['req'];st=z['step'];ids=z['req_ids'];out=[]
    k=st.astype(np.int64)*4096+r;u,inv,c=np.unique(k,return_inverse=True,return_counts=True);dec=c[inv]<=16
    for i,n in enumerate(ids):
        m=r==i
        out.append((str(n),int(m.sum()),int((m&dec).sum())))
    return f,out
with Pool(16) as p:res=p.map(one,fs,chunksize=8)
agg=collections.defaultdict(lambda:[0,0,0])
for f,o in res:
    for n,a,b in o:
        t=n.split('-')[0]+'-'+(n.split('-')[1] if n.startswith(('cmpl-replay','probe')) else '')
        agg[t][0]+=a;agg[t][1]+=b;agg[t][2]+=1
print(dict(agg))
json.dump([(f,o) for f,o in res],open('/data/Jarrel/expert-predict/scan.json','w'))
