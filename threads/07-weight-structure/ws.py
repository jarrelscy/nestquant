"""Thread 07: statistical structure in GLM-5.3 / MiMo-2.6 expert weights (CPU only)."""
import os, json, math, sys
os.environ.setdefault('OMP_NUM_THREADS','16'); os.environ.setdefault('MKL_NUM_THREADS','16')
import numpy as np, torch
torch.set_num_threads(16)
sys.path.insert(0,'/home/coder/git/orbit-duet')
from orbit_duet.source import dequantize
from safetensors import safe_open
from pathlib import Path

GLM='/tmp/orbit-duet-glm53-fp8'; GLM66='/home/coder/git/orbit-duet/runs/source_glm53'
MIMO='/home/coder/git/orbit-duet/runs/source_mimo'
PROJ=['gate_proj','up_proj','down_proj']
_index=None
def glm_expert(L,E):
    global _index
    local=Path(GLM66)/f'layer_{L}'/f'expert_{E}.safetensors'
    out=[]
    for p in PROJ:
        if local.exists():
            with safe_open(str(local),'pt') as f: w=f.get_tensor(p+'.weight'); s=f.get_tensor(p+'.weight_scale_inv')
        else:
            if _index is None: _index=json.load(open(GLM+'/model.safetensors.index.json'))['weight_map']
            k=f'model.layers.{L}.mlp.experts.{E}.{p}'
            assert os.path.exists(GLM+'/'+_index[k+'.weight']+'.verified.json')
            with safe_open(GLM+'/'+_index[k+'.weight'],'pt') as f: w=f.get_tensor(k+'.weight'); s=f.get_tensor(k+'.weight_scale_inv')
        out.append(dequantize(w,s,(128,128)))
    return out
def mimo_raw(L=55,E=70):
    out=[]
    with safe_open(f'{MIMO}/layer_{L}/expert_{E}.safetensors','pt') as f:
        for p in PROJ: out.append((f.get_tensor(p+'.weight'),f.get_tensor(p+'.weight_scale')))
    return out

def had128():
    H=torch.ones(1,1)
    while H.shape[0]<128: H=torch.cat([torch.cat([H,H],1),torch.cat([H,-H],1)],0)
    return H/math.sqrt(128)
H=had128()
def rot_in(w,seed=1):  # block-128 Hadamard with random signs on input (column) dim
    g=torch.Generator().manual_seed(seed); s=(torch.randint(0,2,(w.shape[1],),generator=g)*2-1).float()
    x=(w*s).reshape(w.shape[0],-1,128)@H.T; return x.reshape(w.shape)
def rot_both(w,seed=1): return rot_in(rot_in(w,seed).T.contiguous(),seed+7).T.contiguous()

def kurt(x):
    x=x.double().flatten(); x=x-x.mean(); v=(x*x).mean(); return float((x**4).mean()/v**2-3)
def tails(x):
    x=x.flatten(); s=x.square().mean().sqrt(); a=(x.abs()/s)
    return {f'>{k}s':float((a>k).float().mean()) for k in (3,4,6)}
def spread(v):  # RMS values -> summary
    l=torch.log2(v.double().clamp_min(1e-30))
    return dict(std_log2=float(l.std()), max_over_median=float(v.max()/v.median()), p99_over_p1=float(torch.quantile(v.double(),.99)/torch.quantile(v.double(),.01)))

def rwf_gain(var,R):
    """Gaussian reverse water-filling gain (dB) of variable rate over components with variances var vs uniform rate R."""
    v=var.double().clamp_min(1e-30); lo,hi=math.log(float(v.min()))-60,math.log(float(v.max()))
    for _ in range(200):
        mid=(lo+hi)/2; th=math.exp(mid); r=(0.5*torch.log2(v/th)).clamp_min(0).mean()
        if r>R: lo=mid
        else: hi=mid
    th=math.exp((lo+hi)/2); D=torch.minimum(v,torch.tensor(th,dtype=v.dtype)).mean()
    return float(10*math.log10(float(v.mean())*2**(-2*R)/float(D)))
def amgm_db(var):
    v=var.double().clamp_min(1e-30); return float(10*math.log10(float(v.mean())/math.exp(float(v.log().mean()))))

_LM={}
def lloyd(bits,sample=None,iters=60):
    if sample is None:
        if bits in _LM: return _LM[bits]
        sample=torch.randn(4_000_000,generator=torch.Generator().manual_seed(0))
    x=sample.flatten().double(); n=2**bits
    c=torch.quantile(x[:1_000_000],torch.linspace(.5/n,1-.5/n,n,dtype=torch.float64))
    for _ in range(iters):
        idx=torch.bucketize(x,(c[1:]+c[:-1])/2); s=torch.zeros(n,dtype=torch.float64).index_add_(0,idx,x); k=torch.bincount(idx,minlength=n).double()
        c=torch.where(k>0,s/k.clamp_min(1),c)
    if sample is None or len(_LM)==0 and False: pass
    return c
for b in (2,4): _LM[b]=lloyd(b)
def sq_snr(z,c):
    z=z.flatten().double(); idx=torch.bucketize(z,(c[1:]+c[:-1])/2); e=(z-c[idx]).square().mean()
    return float(10*math.log10(float(z.square().mean())/float(e)))

def scale_study(w):
    """SNR(dB) of a fixed unit-Gaussian Lloyd-Max SQ with (i) per-tensor, (ii) per-row, (iii) per-1x16-block RMS normalisation (scales exact),
    plus per-tensor with a codebook trained on the data marginal; and Gaussian RWF rate-allocation gains over rows / 16-blocks."""
    out={}
    t=w/w.square().mean().sqrt(); r=w/w.square().mean(1,keepdim=True).sqrt()
    bv=w.reshape(w.shape[0],-1,16).square().mean(2,keepdim=True); b=(w.reshape(w.shape[0],-1,16)/bv.sqrt()).reshape(w.shape)
    sub=t.flatten()[torch.randperm(t.numel(),generator=torch.Generator().manual_seed(3))[:2_000_000]]
    for bits in (2,4):
        c=_LM[bits]
        out[f'sq{bits}_tensor']=sq_snr(t,c); out[f'sq{bits}_row']=sq_snr(r,c); out[f'sq{bits}_blk16']=sq_snr(b,c)
        out[f'sq{bits}_tensor_trained']=sq_snr(t,lloyd(bits,sub))
        rv=w.square().mean(1); out[f'rwf{bits}_rows']=rwf_gain(rv,bits); out[f'rwf{bits}_cols']=rwf_gain(w.square().mean(0),bits)
        out[f'rwf{bits}_blk16']=rwf_gain(bv.flatten(),bits)
    return out

def svd_study(w):
    s=torch.linalg.svdvals(w.double()); e=s**2; e=e/e.sum()
    g=torch.randn(w.shape,generator=torch.Generator().manual_seed(5),dtype=torch.float64)
    # control: same row/col RMS profile, i.i.d. Gaussian entries
    rr=w.double().square().mean(1,keepdim=True).sqrt(); cc=w.double().square().mean(0,keepdim=True).sqrt(); cc=cc/cc.square().mean().sqrt()
    sg=torch.linalg.svdvals(g*rr*cc); eg=sg**2; eg=eg/eg.sum()
    m,n=w.shape; res={}
    best=(0,0.0)
    for k in (1,2,4,8,16,32,64,128,256):
        f=float(e[:k].sum()); fg=float(eg[:k].sum())
        db=-10*math.log10(1-f); over=k*(m+n)*8/(m*n)  # 8-bit factors
        net=db/6.02-over; res[k]=dict(energy=f,energy_ctrl=fg,residual_db=db,overhead_bits=over,net_bits=net)
        if net>best[1]: best=(k,net)
    res['best']=best; res['effective_rank_ratio']=float(torch.exp(-(e*e.clamp_min(1e-300).log()).sum())/min(m,n))
    res['effective_rank_ratio_ctrl']=float(torch.exp(-(eg*eg.clamp_min(1e-300).log()).sum())/min(m,n))
    return res

def marg(w):
    return dict(kurtosis=kurt(w),tails=tails(w),row=spread(w.square().mean(1).sqrt()),col=spread(w.square().mean(0).sqrt()),
                top1pct_col_energy=float(w.square().sum(0).sort(descending=True).values[:max(1,w.shape[1]//100)].sum()/w.square().sum()),
                top1pct_row_energy=float(w.square().sum(1).sort(descending=True).values[:max(1,w.shape[0]//100)].sum()/w.square().sum()))

def gate_up(g,u):
    cos=(g*u).sum(1)/(g.norm(dim=1)*u.norm(dim=1)); r2=cos.square()
    # energy of up inside gate's row space vs chance (2048/6144)
    Q,_=torch.linalg.qr(g.double().T); pu=(u.double()@Q); frac=float(pu.square().sum()/u.double().square().sum())
    return dict(mean_cos=float(cos.mean()),mean_abs_cos=float(cos.abs().mean()),mean_r2=float(r2.mean()),max_abs_cos=float(cos.abs().max()),
                cond_gain_bits=float((-0.5*torch.log2(1-r2)).mean()),up_in_gate_rowspace=frac,chance=g.shape[0]/g.shape[1])

def per_expert(ws,control=False):
    out={}
    for p,w in zip(PROJ,ws):
        w=w.float(); wr=rot_both(w); wi=rot_in(w)
        out[p]=dict(pre=marg(w),post_in=marg(wi),post_both=marg(wr),scale_pre=scale_study(w),scale_post_both=scale_study(wr),scale_post_in=scale_study(wi),svd=svd_study(w))
    out['gate_up']=dict(pre=gate_up(ws[0].float(),ws[1].float()),post_both=gate_up(rot_both(ws[0].float()),rot_both(ws[1].float())))
    return out
