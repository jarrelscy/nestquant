"""Full-expert A/B: NestQuant candidate chains vs installed EXL3, same process, interleaved graphs, L2-cold rotation."""
import torch,json,sys;torch.cuda.set_per_process_memory_fraction(12/80)
import torch.nn.functional as F
from nq import *
from timing import bench
from orbit_duet.exl3_adapter import EXL3Expert
NC=6
tuned=json.load(open('tune.json'))
R='/home/coder/git/orbit-duet/runs/glm53_pilot_matched_l16/exl3_e36/'
import os
CH={2:['T2'],3:['T2R1','J3'],4:['T4','T2R2','T2H','J4','J4_inter']} if os.environ.get('J') else {2:['UNI2','T2','H2','H2B'],3:['T2R1'],4:['UNI4','T4','H4','T2R2','T2R2_inter','T2H','T2H_inter']}
def parse(k):
    d=k.split('_')[0];return d,('_inter' not in k),('_lutr' in k)
h16=lambda n:torch.ones(n,device='cuda').half()
sv_g,sv_u,su_d,su_x,sv_d=h16(2048),h16(2048),h16(2048),h16(6144),h16(6144)
out={}
for B in [1,2,3,4]:
    x32=torch.randn(B,6144,device='cuda');xh=torch.empty(B,6144,device='cuda').half()
    acc=torch.zeros(B,4096,device='cuda');hh=torch.empty(B,2048,device='cuda').half()
    y=torch.zeros(B,6144,device='cuda');yo=torch.empty(B,6144,device='cuda').half()
    tb=min(B,4) if B in (1,4) else (1 if B<3 else 4)
    fns={};meta={}
    for bits,names in CH.items():
        for k in names:
            d,sp,lr=parse(k)
            gu=[Proj(d,4096,6144,split=sp,lutr=lr) for _ in range(NC)];dn=[Proj(d,6144,2048,split=sp,lutr=lr) for _ in range(NC)]
            cg=tuple(tuned[f'{k}|gu|{tb}']['cfg']);cd=tuple(tuned[f'{k}|down|{tb}']['cfg'])
            def chain(gu=gu,dn=dn,cg=cg,cd=cd,had=True):
                for i in range(NC):
                    if had:M.had_in(x32,xh,su_x)
                    gu[i](xh,acc,cg);M.swiglu(acc,hh,sv_g,sv_u,su_d,y);dn[i](hh,y,cd)
                    if had:M.had_in(y,yo,sv_d)
            fns[f'{bits}b_{k}']=chain
            fns[f'{bits}b_{k}_gemv_only']=(lambda gu=gu,dn=dn,cg=cg,cd=cd:[(gu[i](xh,acc,cg),dn[i](hh,y,cd)) for i in range(NC)])
    for bits in [2,4]:
        ex=[EXL3Expert(R+f'expert_{bits}.bin') for _ in range(NC)]
        xhh=x32.half();g=torch.empty(B,2048,device='cuda');u=torch.empty(B,2048,device='cuda');h=torch.empty(B,2048,device='cuda',dtype=torch.half);yy=torch.empty(B,6144,device='cuda',dtype=torch.half)
        def kern(e):
            o=e.objects;o[0].bc.run(xhh,g);o[1].bc.run(xhh,u);torch.ops.aten.mul.out(F.silu(g),u,out=g);h.copy_(g);o[2].bc.run(h,yy)
        fns[f'{bits}b_EXL3_adapter']=(lambda ex=ex:[e.forward_batch(x32) for e in ex])
        fns[f'{bits}b_EXL3_kernels']=(lambda ex=ex,kern=kern:[kern(e) for e in ex])
    med,rng=bench(fns,blocks=40,repeats=1)
    out[B]={k:dict(med=v/NC,p10=rng[k][1]/NC) for k,v in med.items()}
    for k in sorted(out[B]):print(B,k,round(out[B][k]['med'],2),round(out[B][k]['p10'],2),flush=True)
    del fns;torch.cuda.empty_cache()
json.dump(out,open('chain_j.json' if os.environ.get('J') else 'chain.json','w'),indent=1)
