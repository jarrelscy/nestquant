import sys, json, torch
from common import *
setup()
model=sys.argv[1]
Wt=teacher_weights(model)
cand={}
for bits in [2,4]:
    parts,ref=load_exl3(model,bits)
    m=Tuned(parts,[])
    Ws,bs=m.dense()
    print(bits,[float((a-b).norm()/b.norm()) for a,b in zip(Ws,ref)])
    cand[f'exl3_{bits}']=(ref,[None]*3); cand[f'mine_{bits}']=(Ws,bs)
    print('Q stats', [ (float(p['Q'].min()),float(p['Q'].max()),float(p['Q'].std())) for p in parts], 'su',[float(p['suh'].abs().mean()) for p in parts])
r=eval_captures(model,Wt,cand)
for k,v in r.items(): print(k, {a:(round(b,3) if isinstance(b,float) else b) for a,b in v.items()})
