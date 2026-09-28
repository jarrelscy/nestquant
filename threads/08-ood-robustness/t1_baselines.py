from common import *
init()
c,rows=capture();res={}
for E in [36,92,165]:
    w=native(E);s=stats(E)
    m=dict(exl3_2=exl3_artifact(E,2),exl3_4=exl3_artifact(E,4),nvfp4=nvfp4(E,w))
    x0,p0=sample(E)
    m['refit4_statsgram']=exl3_fit(w,s['grams'][0],s['grams'][1],4,s['metadata']['training_rows'])
    r={}
    for k,wq in m.items():
        r[k]={d:rel_err(c['x'][idx],w,wq) for d,idx in rows.items()}
        r[k]['train_sample_w']=rel_err(x0,w,wq,weights=p0)
        print(E,k,{a:round(b,3) for a,b in r[k].items()},flush=True)
    r['refit_vs_artifact_weight_diff']=[float((a-b).norm()/b.norm()) for a,b in zip(m['refit4_statsgram'],m['exl3_4'])]
    print(r['refit_vs_artifact_weight_diff'])
    res[E]=r
json.dump(res,open('baselines.json','w'),indent=1)
