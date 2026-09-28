"""Fixed-length refinement to the exact MXFP4 index given a noisy base (Gaussian error, relative MSE D per row)."""
import ws,json,math,torch
torch.set_num_threads(8)
LUT=torch.tensor([0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6],dtype=torch.float64)
out={}
for pi,(packed,scale) in enumerate(ws.mimo_raw()):
    ids=torch.stack([packed&15,packed>>4],-1).flatten(-2).long(); m,n=ids.shape
    s=torch.ldexp(torch.ones(scale.shape,dtype=torch.float64),(scale.long()-127).int()).repeat_interleave(32,-1)
    w=LUT[ids]*s; sig2=w.square().mean(1,keepdim=True)
    prior=torch.bincount(ids.flatten(),minlength=16).double(); prior=(prior/prior.sum()).clamp_min(1e-12)
    g=torch.Generator().manual_seed(2); sub=torch.randperm(m,generator=g)[:256]
    wi,si,ti,s2=w[sub],s[sub],ids[sub],sig2[sub]
    E=float(wi.square().sum()); res={}
    for D in (0.069,0.10,0.15,0.0172,0.025):
        y=wi+torch.randn(wi.shape,generator=g,dtype=torch.float64)*(D*s2).sqrt()
        cand=LUT[None,None,:]*si[...,None]
        ll=-(y[...,None]-cand).square()/(2*D*s2[...,None])+prior.log()
        logp=ll-ll.logsumexp(-1,keepdim=True)
        Hc=float(-(logp.gather(-1,ti[...,None]).squeeze(-1)).mean()/math.log(2))
        order=logp.argsort(-1,descending=True); r={'H_cond_bits':Hc,'base_snr_db':10*math.log10(1/D)}
        for bits in (1,2):
            k=2**bits; top=order[...,:k]; hit=(top==ti[...,None]).any(-1)
            # decode: true value if hit else best candidate (MAP among top-k = top[0])
            dec=torch.where(hit,wi,cand.gather(-1,top[...,:1]).squeeze(-1))
            err=float((dec-wi).square().sum()); r[f'fixed{bits}b']=dict(exact_fraction=float(hit.double().mean()),snr_db=10*math.log10(E/max(err,1e-300)))
        res[str(D)]=r
    out[ws.PROJ[pi]]=res; print(ws.PROJ[pi],json.dumps(res),flush=True)
json.dump(out,open('fp4_refine.json','w'),indent=1)
