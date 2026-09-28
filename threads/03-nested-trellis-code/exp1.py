# Residual-stage successive refinement using the EXL3 kernel (mul1 codebook, L=16, tail-biting 256 tiles)
import torch, json, math, sys
torch.cuda.set_per_process_memory_fraction(12/80)
from exllamav3.modules.quant.exl3_lib.quantize import quantize_tiles
torch.manual_seed(0)
N=2048
x=torch.randn(N,256,device='cuda')
CB=sys.argv[1] if len(sys.argv)>1 else 'mul1'
def Q(t,K,s):
    qa={"K":K}
    if CB!='3inst': qa[CB]=True
    q,_=quantize_tiles((t*s).contiguous(),qa); return q/s
def best_scale(t,K,grid):
    best=None
    for s in grid:
        q=Q(t,K,s); m=((t-q)**2).mean().item()
        if best is None or m<best[0]: best=(m,s,q)
    return best
def db(m,R): return 10*math.log10(m/2**(-2*R))
out={}
def rep(name,m,R):
    out[name]=(m,db(m,R)); print(f"{name:40s} mse={m:.6f}  {db(m,R):+.3f} dB vs RD",flush=True)
# native
nat={}
for K in [2,3,4]:
    m,s,q=best_scale(x,K,[0.85+0.025*i for i in range(16)])
    nat[K]=(m,s); rep(f"native_K{K} (s={s:.3f})",m,K)
m2,s2,q2=nat[2][0],nat[2][1],Q(x,2,nat[2][1])
e2=x-q2
print("base residual std",e2.std().item(), "kurtosis",((e2/e2.std())**4).mean().item())
# stage K=2 on base residual (level 4 two-stage)
def rs(t,K):
    sd=t.std().item()
    return best_scale(t,K,[(0.7+0.05*i)/sd for i in range(14)])
m,s,r4=rs(e2,2); rep("base+K2res -> L4",m,4)
# stage K=1 -> L3, then K=1 -> L4
m,s,r3=rs(e2,1); rep("base+K1res -> L3",m,3)
e3=e2-r3
m,s,r4b=rs(e3,1); rep("base+K1+K1 -> L4",m,4)
print("L3 residual std",e3.std().item(),"kurt",((e3/e3.std())**4).mean().item())
json.dump(out,open(f'exp1_{CB}.json','w'),indent=1)
