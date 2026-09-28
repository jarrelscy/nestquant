"""Empirical entropy of EXL3 trellis symbols in matched .bin artifacts."""
import sys,json,math,glob,torch
sys.path.insert(0,'/home/coder/git/orbit-duet'); torch.set_num_threads(4)
from orbit_duet.exl3_adapter import read_legacy
def H(counts):
    c=counts.double(); n=c.sum(); p=c[c>0]/n; h=float(-(p*p.log2()).sum())
    return h+float((c>0).sum()-1)/(2*float(n)*math.log(2))  # Miller-Madow
def condH(joint,ctxbins):  # joint indexed ctx*nsym+sym
    j=joint.double().reshape(ctxbins,-1); return H(joint)-H(j.sum(1))
files=sorted(glob.glob('/home/coder/git/orbit-duet/runs/glm53_pilot_matched_l*/exl3_e*/expert_*.bin'))+sorted(glob.glob('/home/coder/git/orbit-duet/runs/full55_exl3_e70/expert_*.bin'))
out={}
for f in files:
    K=int(f[-5]); rec=[]
    for i,v in enumerate(read_legacy(f,'cpu')):
        t=v['trellis'].view(torch.int16).to(torch.int32)&0xFFFF  # [tr,tc,16K] uint16 words
        words=t.flatten()
        nsym=16//K; mask=(1<<K)-1
        sym=torch.stack([(t>>(K*j))&mask for j in range(nsym)],-1).reshape(t.shape[0],t.shape[1],-1)  # symbols of a tile, word-major
        s=sym.flatten()
        h0=H(torch.bincount(s,minlength=1<<K))
        a=sym[...,:-1].flatten(); b=sym[...,1:].flatten(); h1=condH(torch.bincount(a*(1<<K)+b,minlength=1<<(2*K)),1<<K)
        a2=sym[...,:-2].flatten(); a1=sym[...,1:-1].flatten(); b2=sym[...,2:].flatten()
        h2=condH(torch.bincount((a2*(1<<K)+a1)*(1<<K)+b2,minlength=1<<(3*K)),1<<(2*K))
        # position-in-tile dependence
        pos=torch.arange(sym.shape[-1]).expand_as(sym).flatten(); hp=condH(torch.bincount(pos*(1<<K)+s,minlength=sym.shape[-1]<<K),sym.shape[-1])
        hw=H(torch.bincount(words,minlength=65536))
        rec.append(dict(proj=i,K=K,symbol_H0=h0,symbol_H_given_prev=h1,symbol_H_given_prev2=h2,symbol_H_given_pos=hp,word16_H=hw,n_words=int(words.numel())))
    out[f.replace('/home/coder/git/orbit-duet/runs/','')]=rec; print(f,[(round(r['symbol_H0'],4),round(r['symbol_H_given_prev2'],4),round(r['word16_H'],3)) for r in rec],flush=True)
json.dump(out,open('trellis_entropy.json','w'),indent=1)
