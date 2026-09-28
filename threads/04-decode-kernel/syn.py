"""Decode-ops budget: full-expert chain with synthetic extra ops/weight (SYN = T-mul1 + R extra int ops/weight) vs EXL3."""
import torch,json;torch.cuda.set_per_process_memory_fraction(12/80)
import torch.nn.functional as F
from nq import *
from timing import bench
from orbit_duet.exl3_adapter import EXL3Expert
NC=6;tuned=json.load(open('tune.json'))
R_='/home/coder/git/orbit-duet/runs/glm53_pilot_matched_l16/exl3_e36/'
h16=lambda n:torch.ones(n,device='cuda').half()
sv_g,sv_u,su_d,su_x,sv_d=h16(2048),h16(2048),h16(2048),h16(6144),h16(6144)
out={}
for B in [1,4]:
    x32=torch.randn(B,6144,device='cuda');xh=torch.empty(B,6144,device='cuda').half()
    acc=torch.zeros(B,4096,device='cuda');hh=torch.empty(B,2048,device='cuda').half()
    y=torch.zeros(B,6144,device='cuda');yo=torch.empty(B,6144,device='cuda').half()
    fns={}
    for bits,d,ref in [(4,'SYN4','T4'),(2,'SYN2','T2')]:
        cg=tuple(tuned[f'{ref}|gu|{B}']['cfg']);cd=tuple(tuned[f'{ref}|down|{B}']['cfg'])
        for r in [0,2,4,8,12,16]:
            gu=[Proj(d,4096,6144,r=r) for _ in range(NC)];dn=[Proj(d,6144,2048,r=r) for _ in range(NC)]
            def chain(gu=gu,dn=dn,cg=cg,cd=cd):
                for i in range(NC):
                    M.had_in(x32,xh,su_x);gu[i](xh,acc,cg);M.swiglu(acc,hh,sv_g,sv_u,su_d,y);dn[i](hh,y,cd);M.had_in(y,yo,sv_d)
            fns[f'{bits}b_SYN_R{r}']=chain
        ex=[EXL3Expert(R_+f'expert_{bits}.bin') for _ in range(NC)]
        fns[f'{bits}b_EXL3_adapter']=(lambda ex=ex:[e.forward_batch(x32) for e in ex])
    med,rng=bench(fns,blocks=40,repeats=1)
    out[B]={k:v/NC for k,v in med.items()}
    for k in sorted(out[B]):print(B,k,round(out[B][k],2),flush=True)
    del fns;torch.cuda.empty_cache()
json.dump(out,open('syn.json','w'),indent=1)
