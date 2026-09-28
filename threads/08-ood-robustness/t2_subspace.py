"""Part 1: where do OOD inputs lie relative to the calibration Hessian, and where is the EXL3-4 error."""
from common import *
init()
c,rows=capture();out={}
bands=[(0,.01),(.01,.05),(.05,.2),(.2,.5),(.5,1.)]
for E in [36,92,165]:
    w=native(E);x0,p0=sample(E);s=stats(E);q=exl3_artifact(E,4);nv=nvfp4(E,w)
    nr=s['metadata']['source']['routed_rows']
    sets={'train_routed':(x0[:nr],None),'train_context':(x0[nr:],None),'train_weighted':(x0,p0)}
    for k in ['ID','OOD','ood:fasta','ood:encoded_bytes','ood:smt_bitvectors','ood:scientific_telemetry']:sets[k]=(c['x'][rows[k]],None)
    res={}
    neff=float(p0.square().sum()**2/p0.pow(4).sum())
    res['n_eff_weighted_rows']=neff
    for proj,(HH,Wm,Qs) in {'gate':(s['grams'][0],w[0],{'exl3_4':q[0],'nvfp4':nv[0]}),'down':(s['grams'][1],w[2],{'exl3_4':q[2],'nvfp4':nv[2]})}.items():
        H=HH.cuda().double();ev,U=torch.linalg.eigh(H);ev=ev.flip(0);U=U.flip(1);n=len(ev)
        cum=ev.cumsum(0)/ev.sum()
        k99=int((cum<.99).sum())+1;k999=int((cum<.999).sum())+1
        r=dict(k99=k99,k999=k999,cond_top_over_median=float(ev[0]/ev[n//2]))
        # weight / error energy per eigendirection
        wk=(Wm.double()@U).square().sum(0)
        eks={m:((Qm.double()-Wm.double())@U).square().sum(0) for m,Qm in Qs.items()}
        r['err_gain_per_band']={m:[float(ek[int(a*n):int(b*n)].sum()/wk[int(a*n):int(b*n)].sum()) for a,b in bands] for m,ek in eks.items()}
        for name,(xs,pw) in sets.items():
            xs=xs.cuda().float()
            if proj=='down':xs=torch.cat([hidden(xs[i:i+2048],w) for i in range(0,len(xs),2048)])
            if pw is not None:xs=xs*pw.cuda()[:,None]
            z=(xs.double()@U).square()      # per-token energy per eigendir
            zt=z.sum(0);tot=zt.sum()
            d=dict(energy_band=[float(zt[int(a*n):int(b*n)].sum()/tot) for a,b in bands],
                   energy_outside_99pct=float(zt[k99:].sum()/tot),energy_outside_999pct=float(zt[k999:].sum()/tot),
                   alignment=float((zt*ev).sum()/tot/(ev.mean())))
            for m,ek in eks.items():
                contrib=zt*ek  # diagonal approx of ||E x||^2
                d[m+'_errshare_band']=[float(contrib[int(a*n):int(b*n)].sum()/contrib.sum()) for a,b in bands]
                d[m+'_errshare_outside_99pct']=float(contrib[k99:].sum()/contrib.sum())
                # exact linear pre-activation relative error
                Em=(Qs[m].double()-Wm.double())
                d[m+'_linear_rel']=float(((xs.double()@Em.T).square().sum()/(xs.double()@Wm.double().T).square().sum())**.5*100)
            r[name]=d
        res[proj]=r
        print(E,proj,'k99',k99,'k999',k999,json.dumps({k:(round(v['energy_outside_99pct'],4),round(v['alignment'],2),[round(t,3) for t in v['exl3_4_errshare_band']],round(v['exl3_4_linear_rel'],2),round(v['nvfp4_linear_rel'],2)) for k,v in r.items() if isinstance(v,dict) and 'alignment' in v}),'\n gain',{m:[round(t,3) for t in g] for m,g in r['err_gain_per_band'].items()},flush=True)
    out[E]=res
json.dump(out,open('subspace.json','w'),indent=1)
