"""Pre-registered selection on TRAINING held-out only (never the capture).
Score = mean over experts of mean(val_w, val_u, val_tail) each relative to that expert's baseline.
Constraint: val_w (router-weighted held-out ID) must not exceed baseline on ANY expert."""
import json,sys
bits=sys.argv[1] if len(sys.argv)>1 else '4'
R=json.load(open(f'sweep_select_{bits}.json'));E=['36','92','165']
names=sorted({k.split('/')[1] for k in R if '#' not in k})
rows=[]
for n in names:
    if not all(f'{e}/{n}' in R for e in E):continue
    rel={m:sum(R[f'{e}/{n}'][m]/R[f'{e}/baseline'][m] for e in E)/3 for m in ['val_w','val_u','val_tail','fit_w']}
    ok=all(R[f'{e}/{n}']['val_w']<=R[f'{e}/baseline']['val_w'] for e in E)
    rows.append((sum(rel[m] for m in ['val_w','val_u','val_tail'])/3,n,ok,rel))
rows.sort()
for s,n,ok,rel in rows:print(f'{n:28s} score {s:.4f} ok={ok} '+' '.join(f'{m} {v:.4f}' for m,v in rel.items()))
best=[r for r in rows if r[2]][0]
json.dump(dict(bits=bits,selected=best[1],score=best[0],rule=__doc__,table=[dict(name=n,score=s,ok=ok,**rel) for s,n,ok,rel in rows]),open(f'selection_{bits}.json','w'),indent=1)
print('SELECTED',best[1])
