# Joint alternating refit alpha*D2 + beta*D4 (2-level) and a2*D2+a3*D3+a4*D4 (3-level, K1+K1 stages)
import torch, json, math
torch.cuda.set_per_process_memory_fraction(12/80)
from exllamav3.modules.quant.exl3_lib.quantize import quantize_tiles
torch.manual_seed(0)
N=2048
x=torch.randn(N,256,device='cuda')
def Q(t,K,s):
    q,_=quantize_tiles((t*s).contiguous(),{"K":K,"mul1":True}); return q/s
def mse(a): return (a**2).mean().item()
def db(m,R): return 10*math.log10(m/2**(-2*R))
def rs(t,K,center=None):
    sd=t.std().item(); best=None
    grid=[(0.8+0.04*i)/sd for i in range(11)] if center is None else [center*(0.94+0.02*i) for i in range(7)]
    for s in grid:
        q=Q(t,K,s); m=mse(t-q)
        if best is None or m<best[0]: best=(m,s,q)
    return best
S2=1.0
out={}
# 2-level
for (a,b) in [(1,0),(1,1),(1,2),(1,4)]:
    q2=Q(x,2,S2); m,s4,r=rs(x-q2,2)
    hist=[]
    for it in range(4):
        if b>0:
            y=x-(b/(a+b))*r
            q2=Q(y,2,S2)
        m,s4,r=rs(x-q2,2,s4 if it>0 else None)
        D2=mse(x-q2); D4=mse(x-q2-r)
        hist.append((D2,D4)); print(f"a={a} b={b} it={it} D2={D2:.6f} ({db(D2,2):+.3f}dB) D4={D4:.6f} ({db(D4,4):+.3f}dB)",flush=True)
        if b==0: break
    out[f"2lvl_a{a}_b{b}"]=hist
# 3-level K1+K1
for (a2,a3,a4) in [(1,0,0),(1,1,1),(1,0.5,1),(1,1,2)]:
    q2=Q(x,2,S2); m,s3,r3=rs(x-q2,1); m,s4,r4=rs(x-q2-r3,1)
    hist=[]
    for it in range(4):
        if a3+a4>0:
            y=x-(a3*r3+a4*(r3+r4))/(a2+a3+a4); q2=Q(y,2,S2)
            t3=(x-q2)-a4*r4/(a3+a4)
            m,s3,r3=rs(t3,1,s3)
        m,s4,r4=rs(x-q2-r3,1,s4)
        D2=mse(x-q2);D3=mse(x-q2-r3);D4=mse(x-q2-r3-r4)
        hist.append((D2,D3,D4)); print(f"3lvl {a2},{a3},{a4} it={it} D2={D2:.6f}({db(D2,2):+.3f}) D3={D3:.6f}({db(D3,3):+.3f}) D4={D4:.6f}({db(D4,4):+.3f})",flush=True)
        if a3+a4==0: break
    out[f"3lvl_{a2}_{a3}_{a4}"]=hist
json.dump(out,open('exp2_joint.json','w'),indent=1)
