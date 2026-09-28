"""Part 2: fit-time Hessian regularisation sweep with EXL3's own quantizer.
mode=select: fit on 80% of the training sample, score on the 20% held-out (training data only).
mode=final : fit on 100% of the training sample, score on the frozen capture (ID control + OOD)."""
import sys,time
from common import *
init()
mode=sys.argv[1];bits=int(sys.argv[2]);names=sys.argv[3].split(',') if len(sys.argv)>3 else None
EXP=[36,92,165]
def nt(G):return G/G.diagonal().mean()
def rowhash(x):
    v=x.view(torch.int16).double().cuda();r=torch.arange(v.shape[1],device='cuda',dtype=torch.float64)
    return (v*torch.sin(r*1.2345+.1)).sum(1).cpu()
@torch.no_grad()
def kmeans(x,k=8,it=25,seed=0):
    g=torch.Generator().manual_seed(seed);z=x.cuda().float();z=z/z.norm(dim=1,keepdim=True)
    C=z[torch.randperm(len(z),generator=g)[:k].cuda()]
    for _ in range(it):
        a=(z@C.T).argmax(1)
        for j in range(k):
            m=a==j
            if m.any():C[j]=z[m].mean(0);C[j]/=C[j].norm()
    return a.cpu()
def settings():
    S=[('baseline',dict(src='w',sigma=.03))]
    for s in [.003,.01,.1,.3,1.,3.]:S.append((f'damp{s}',dict(src='w',sigma=s)))
    for a in [.1,.25,.5,.75,1.]:S.append((f'unif{a}',dict(src='w',mix=('u',a),sigma=.03)))
    for a in [.25,.5,1.]:S.append((f'layer{a}',dict(src='w',mix=('l',a),sigma=.03)))
    for a in [.1,.3]:S.append((f'diag{a}',dict(src='w',diag=a,sigma=.03)))
    for a in [.5,1.]:S.append((f'groupbal{a}',dict(src='w',mix=('gb',a),sigma=.03)))
    S.append(('unif0.5_damp0.1',dict(src='w',mix=('u',.5),sigma=.1)))
    S.append(('unif0.25_damp0.1',dict(src='w',mix=('u',.25),sigma=.1)))
    S.append(('layer0.5_damp0.1',dict(src='w',mix=('l',.5),sigma=.1)))
    S.append(('minimax',dict(src='mm',sigma=.03)))
    for sg in [.2,.5]:S.append((f'damp{sg}',dict(src='w',sigma=sg)))
    S.append(('damp_gu0.3_d0.1',dict(src='w',sigma=(.3,.1))));S.append(('damp_gu0.3_d1.0',dict(src='w',sigma=(.3,1.))))
    S.append(('oas',dict(src='w',oas=True)))
    S.append(('unif0.5_damp0.3',dict(src='w',mix=('u',.5),sigma=.3)))
    S.append(('unif0.25_damp0.3',dict(src='w',mix=('u',.25),sigma=.3)))
    S.append(('layer0.5_damp0.3',dict(src='w',mix=('l',.5),sigma=.3)))
    S.append(('diag0.3_damp0.3',dict(src='w',diag=.3,sigma=.3)))
    S.append(('damp_gu0.5_d1.0',dict(src='w',sigma=(.5,1.))));S.append(('damp_gu1.0_d3.0',dict(src='w',sigma=(1.,3.))))
    S.append(('unif0.5_damp_gu0.5_d1.0',dict(src='w',mix=('u',.5),sigma=(.5,1.))))
    S.append(('unif0.75_damp_gu0.5_d1.0',dict(src='w',mix=('u',.75),sigma=(.5,1.))))
    S.append(('unif0.5_damp1.0',dict(src='w',mix=('u',.5),sigma=1.)))
    S.append(('layer0.5_damp_gu0.5_d1.0',dict(src='w',mix=('l',.5),sigma=(.5,1.))))
    # router-probability^1 row weighting (thread 06) instead of p^2
    S.append(('p1_damp0.03',dict(src='w',base='p1',sigma=.03)))
    S.append(('p1_damp0.3',dict(src='w',base='p1',sigma=.3)))
    S.append(('p1_damp_gu0.5_d1.0',dict(src='w',base='p1',sigma=(.5,1.))))
    S.append(('p1_unif0.5_damp_gu0.5_d1.0',dict(src='w',base='p1',mix=('u',.5),sigma=(.5,1.))))
    S.append(('p1_unif0.25_damp_gu0.5_d1.0',dict(src='w',base='p1',mix=('u',.25),sigma=(.5,1.))))
    return S
def build(parts,cfg):
    out=[]
    for j in range(2):
        H=nt(parts[cfg.get('base','w')][j])
        if 'mix' in cfg:
            k,a=cfg['mix'];H=(1-a)*H+a*nt(parts[k][j])
        if 'diag' in cfg:a=cfg['diag'];H=(1-a)*H+a*torch.diag(H.diagonal())
        out.append(H)
    return out
results_path=f'sweep_{mode}_{bits}{os.environ.get("TAG","")}.json'
R=json.load(open(results_path)) if os.path.exists(results_path) else {}
for E in EXP:
    t0=time.time();w=native(E);x,p=sample(E);N=len(x)
    perm=torch.randperm(N,generator=torch.Generator().manual_seed(1234))
    if mode=='select':fit,val=perm[:int(.8*N)].sort().values,perm[int(.8*N):].sort().values
    else:fit,val=torch.arange(N),None
    xf,pf=x[fit],p[fit]
    parts={'w':grams(xf,pf,w),'u':grams(xf,torch.ones(len(xf)),w),'p1':grams(xf,pf.sqrt(),w)}
    # layer prior: other experts' training rows (unweighted), deduped against this expert's held-out rows
    others=torch.cat([sample(o)[0] for o in EXP if o!=E]+[xf])
    if val is not None:
        hv=set(rowhash(x[val]).tolist());keep=torch.tensor([h not in hv for h in rowhash(others).tolist()])
        others=others[keep]
    parts['l']=grams(others,torch.ones(len(others)),w);del others
    # group-balanced prior: k-means token groups on fit rows, each group's weighted gram normalised
    lab=kmeans(xf);gb=[torch.zeros(6144,6144,device='cuda'),torch.zeros(2048,2048,device='cuda')];G={}
    for j in range(8):
        m=lab==j
        if m.sum()<32:continue
        gg=grams(xf[m],pf[m],w);G[j]=gg
        for t in range(2):gb[t]+=nt(gg[t])
    parts['gb']=gb
    # tail-val proxy: held-out rows with most gate-input energy outside the fit-H 99% subspace
    if val is not None:
        ev,U=torch.linalg.eigh(parts['w'][0].double());ev=ev.flip(0);U=U.flip(1);k99=int((ev.cumsum(0)/ev.sum()<.99).sum())+1
        xv=x[val].cuda().double();frac=((xv@U[:,k99:]).square().sum(1)/xv.square().sum(1)).cpu();del xv,U
        tail=val[frac>=frac.quantile(.8)]
        evalsets=dict(val_w=(x[val],p[val]),val_u=(x[val],None),val_tail=(x[tail],None),fit_w=(xf[:6000],pf[:6000]))
    else:
        c,rows=capture();evalsets={d:(c['x'][idx],None) for d,idx in rows.items()};evalsets['all']=(c['x'],None)
        # actually routed, router weighted
        rr,sl=torch.where(c['ids']==E);pr=c['p'][rr,sl]
        for grp in ['ID','OOD']:
            m=torch.isin(rr,rows[grp]);evalsets['routed_'+grp]=(c['x'][rr[m]],pr[m])
        evalsets['routed_all']=(c['x'][rr],pr)
    print(E,'prep',time.time()-t0,flush=True)
    for name,cfg in settings():
        if names and name not in names:continue
        key=f'{E}/{name}'
        if key in R:continue
        t=time.time()
        if cfg['src']=='mm':
            # minimax-ish: multiplicative weights over token groups using in-fit per-group error of the baseline fit
            H=build(parts,dict(src='w'));q=exl3_fit(w,H[0],H[1],bits,1,sigma=.03)
            errs={j:rel_err(xf[lab==j],w,q,weights=pf[lab==j]) for j in G}
            mx=max(errs.values());wt={j:(errs[j]/mx)**4 for j in G}
            HG=sum(wt[j]*nt(G[j][0]) for j in G);HD=sum(wt[j]*nt(G[j][1]) for j in G);H=[.5*nt(parts['w'][0])+.5*nt(HG),.5*nt(parts['w'][1])+.5*nt(HD)]
        else:H=build(parts,cfg)
        sigma=cfg.get('sigma')
        if cfg.get('oas'):
            # Oracle-approximating shrinkage toward (mean diag) I with the router-weighted effective sample size
            neff=float(pf.double().square().sum()**2/pf.double().pow(4).sum());sig=[]
            for Hj in H:
                S=Hj.double();P=S.shape[0];tr=float(S.trace());tr2=float((S*S).sum())
                rho=min(1.,((1-2/P)*tr2+tr*tr)/((neff+1-2/P)*(tr2-tr*tr/P)));sig.append(rho/(1-rho) if rho<1 else 1e3)
            sigma=tuple(sig);print('oas neff',neff,'sigma',sigma,flush=True)
        q=exl3_fit(w,H[0],H[1],bits,1,sigma=sigma)
        R[key+'#sigma']=sigma
        R[key]={k:rel_err(xx,w,q,weights=pp) for k,(xx,pp) in evalsets.items()}
        print(key,{k:round(v,3) for k,v in R[key].items()},round(time.time()-t,1),flush=True)
        json.dump(R,open(results_path,'w'),indent=1)
    del parts;torch.cuda.empty_cache()
