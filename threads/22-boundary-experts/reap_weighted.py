# Token-weighted REAP: S_e = sum_t w_t p_te ||y_te|| / sum_t w_t, w_t = 1 + (W-1)*[t within 32 before a boundary].
# sal.npy categories are disjoint boundary buckets plus 'all' (all fit rows), so S_e is a linear combination of existing sums.
# Fixed set = top 26 (10%) by S_e. Report coverage (share of routes landing in the set) on all tokens vs boundary tokens.
import numpy as np, json, os, sys
root=sys.argv[1] if len(sys.argv)>1 else '/tmp/nestquant/19-capture-glmfmt'
Ws=[1,3,10,50]; K=26; out={}
for L in range(3,78):
    p=f'{root}/stats0/L{L}/sal.npy'
    if not os.path.exists(p): continue
    s=np.load(p); allr=s[:,0,:]; bnd=s[:,1:,:].sum(1); th=s[:,1:5,:].sum(1); en=s[:,5:9,:].sum(1)
    ntok=allr[:,0].sum()/8; nb=bnd[:,0].sum()/8
    r={'bnd_token_share':float(nb/ntok)}
    for W in Ws:
        S=allr[:,4]+(W-1)*bnd[:,4]; top=np.argsort(-S)[:K]
        cov=lambda a: float(a[top,0].sum()/max(a[:,0].sum(),1))
        r[W]=dict(top=top.tolist(),cov_all=cov(allr),cov_think=cov(th),cov_end=cov(en),
                  bnd_weight_share=float(W*nb/(ntok+(W-1)*nb)))
    base=set(r[1]['top']); 
    for W in Ws[1:]: r[W]['changed_vs_W1']=len(set(r[W]['top'])-base)
    out[L]=r
json.dump(out,open(f'/tmp/nestquant/bnd-experts/reap_weighted_{os.path.basename(root)}.json','w'))
Ls=sorted(out); print(root,'layers',len(Ls),'boundary token share %.3f'%np.mean([out[L]['bnd_token_share'] for L in Ls]))
print(' W  bndWeight  changed/26  cov_all  cov_think  cov_end')
for W in Ws:
    f=lambda k: np.mean([out[L][W][k] for L in Ls])
    print(f"{W:3d}  {f('bnd_weight_share'):.2f}      {f('changed_vs_W1') if W>1 else 0:5.1f}     {f('cov_all'):.3f}   {f('cov_think'):.3f}     {f('cov_end'):.3f}")
