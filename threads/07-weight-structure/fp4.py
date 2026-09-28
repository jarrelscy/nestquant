"""MiMo MXFP4 source-grid structure: index entropy, nested prefix codes on the e2m1 grid, lossless-refinement rate given a trellis-grade base."""
import ws,json,math,torch
torch.set_num_threads(8)
LUT=torch.tensor([0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6],dtype=torch.float64)
MAG=torch.tensor([0,.5,1,1.5,2,3,4,6],dtype=torch.float64)
def H(c):
    c=c.double(); p=c[c>0]/c.sum(); return float(-(p*p.log2()).sum())
def condH(j,nctx): j=j.double().reshape(nctx,-1); return H(j.flatten())-H(j.sum(1))
out={}
for pi,(packed,scale) in enumerate(ws.mimo_raw()):
    p=ws.PROJ[pi]; r={}
    ids=torch.stack([packed&15,packed>>4],-1).flatten(-2).long()   # [m,n]
    m,n=ids.shape; e=scale.long()                                   # [m,n/32]
    s=torch.ldexp(torch.ones(e.shape,dtype=torch.float64),(e-127).int()).repeat_interleave(32,-1)
    mag=ids&7; sign=ids>>3
    w=LUT[ids]*s; E=float(w.square().sum())
    r['nibble_hist']=torch.bincount(ids.flatten(),minlength=16).tolist()
    r['H_nibble']=H(torch.bincount(ids.flatten(),minlength=16)); r['H_mag']=H(torch.bincount(mag.flatten(),minlength=8)); r['H_sign']=H(torch.bincount(sign.flatten(),minlength=2))
    a=mag[:,:-1].flatten(); b=mag[:,1:].flatten(); r['H_mag_given_prev']=condH(torch.bincount(a*8+b,minlength=64),8)
    pos=(torch.arange(n)%32).expand(m,n).flatten(); r['H_mag_given_pos32']=condH(torch.bincount(pos*8+mag.flatten(),minlength=256),32)
    bm=mag.reshape(m,-1,32).max(2).values; r['block_max_mag_hist']=torch.bincount(bm.flatten(),minlength=8).tolist()
    # magnitude conditional on block max (decoder learns max as side info)
    ctx=bm.repeat_interleave(32,-1).flatten(); r['H_mag_given_blockmax']=condH(torch.bincount(ctx*8+mag.flatten(),minlength=64),8)
    r['H_blockmax']=H(torch.bincount(bm.flatten(),minlength=8))
    r['H_scale']=H(torch.bincount(e.flatten(),minlength=256))
    r['H_scale_given_prev']=condH(torch.bincount(e[:,:-1].flatten()*256+e[:,1:].flatten(),minlength=65536),256)
    r['H_scale_given_rowmode']=None
    r['lossless_order0_bpw']=r['H_nibble']+r['H_scale_given_prev']/32
    r['raw_bpw']=4.25
    # nested/non-nested scalar prefix codes on the magnitude grid (sign kept), centroid recon with scale^2 weights
    s2=s.square().flatten(); mg=MAG[mag.flatten()]
    def cells_err(bounds):  # bounds: list of (lo,hi) index ranges over MAG
        err=0.
        for lo,hi in bounds:
            sel=(mag.flatten()>=lo)&(mag.flatten()<=hi)
            if sel.any():
                c=float((s2[sel]*mg[sel]).sum()/s2[sel].sum()); err+=float((s2[sel]*(mg[sel]-c).square()).sum())
        return err
    def snr(err): return 10*math.log10(E/err)
    best2=max(((k,snr(cells_err([(0,k),(k+1,7)]))) for k in range(7)),key=lambda t:t[1])
    r['prefix2_nested_4_4']=snr(cells_err([(0,3),(4,7)]))
    r['prefix2_best']=dict(split_after=best2[0],snr_db=best2[1])
    import itertools
    best3=max(((c,snr(cells_err([(0,c[0]),(c[0]+1,c[1]),(c[1]+1,c[2]),(c[2]+1,7)]))) for c in itertools.combinations(range(7),3)),key=lambda t:t[1])
    r['prefix3_nested_balanced']=snr(cells_err([(0,1),(2,3),(4,5),(6,7)]))
    r['prefix3_best']=dict(splits=best3[0],snr_db=best3[1])
    # lossless refinement rate given a base with Gaussian error of relative MSE D (per-row sigma): H(idx | y), y=w+N(0,D*sigma_row^2)
    sig2=w.square().mean(1,keepdim=True)
    prior=torch.bincount(ids.flatten(),minlength=16).double(); prior=prior/prior.sum()
    g=torch.Generator().manual_seed(1)
    sub=torch.randperm(m,generator=g)[:256]
    for name,D in [('trellis2_D0.069',0.069),('trellis3_D0.0172',0.0172),('trellis4_D0.0043',0.0043)]:
        wi=w[sub]; si=s[sub]; yi=wi+torch.randn(wi.shape,generator=g,dtype=torch.float64)*(D*sig2[sub]).sqrt()
        cand=LUT[None,None,:]*si[...,None]  # [r,n,16]
        ll=-(yi[...,None]-cand).square()/(2*D*sig2[sub][...,None])+prior.log()[None,None]
        # merge +0/-0 (identical values): treat as distinct symbols anyway (source stores both)
        logp=ll-ll.logsumexp(-1,keepdim=True); true=ids[sub]
        r[f'H_idx_given_base_{name}']=float(-(logp.gather(-1,true[...,None]).squeeze(-1)).mean()/math.log(2))
    out[p]=r; print(p,json.dumps({k:v for k,v in r.items() if 'hist' not in k}),flush=True)
json.dump(out,open('fp4.json','w'),indent=1)
