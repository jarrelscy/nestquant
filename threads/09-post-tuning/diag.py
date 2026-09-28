import torch,sys
from common import *
setup()
model=sys.argv[1]
c=CFG[model]; Wt=teacher_weights(model)
s=torch.load(c['sample'],mmap=True,weights_only=True)
p=s['p'].float()
print('p stats', p.min(), p.max(), (p==p[-1]).float().mean())
sets={'high_p':(p>0.06).nonzero().flatten(),'low_p':(p<=0.06).nonzero().flatten()}
for bits in [2,4]:
    _,ref=load_exl3(model,bits)
    for nm,ids in sets.items():
        num=den=0.
        for i in range(0,len(ids),2048):
            x=s['x'][ids[i:i+2048]].cuda(); t=expert(x,Wt).double(); q=expert(x,ref).double()
            num+=float((q-t).square().sum()); den+=float(t.square().sum())
        print(bits,nm,len(ids),'unweighted rel %.2f'%(100*(num/den)**.5))
