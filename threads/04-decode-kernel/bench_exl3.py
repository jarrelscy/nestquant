import torch,json,sys
torch.cuda.set_per_process_memory_fraction(12/80)
import torch.nn.functional as F
from orbit_duet.exl3_adapter import EXL3Expert
from timing import bench
R='/home/coder/git/orbit-duet/runs/glm53_pilot_matched_l16/exl3_e36/'
NCOPY=6  # 6 copies x 19MB > 40MB L2 -> cold weights when rotating
out={}
for bits in [2,4]:
    ex=[EXL3Expert(R+f'expert_{bits}.bin') for _ in range(NCOPY)]
    for B in [1,2,3,4]:
        x=torch.randn(B,6144,device='cuda').half()
        g=torch.empty(B,2048,device='cuda');u=torch.empty(B,2048,device='cuda');h=torch.empty(B,2048,device='cuda',dtype=torch.half);y=torch.empty(B,6144,device='cuda',dtype=torch.half)
        def kern(e):
            o=e.objects;o[0].bc.run(x,g);o[1].bc.run(x,u);torch.ops.aten.mul.out(F.silu(g),u,out=g);h.copy_(g);o[2].bc.run(h,y)
        def only3(e):
            o=e.objects;o[0].bc.run(x,g);o[1].bc.run(x,u);o[2].bc.run(h,y)
        st={'i':0}
        def rot(fn):
            def f():
                for e in ex: fn(e)   # one call = NCOPY experts, divide later
            return f
        def rep(fn):
            def f():
                for e in ex: fn(ex[0])
            return f
        fns={'adapter_warm':rep(lambda e:e.forward_batch(x)),
             'kern_warm':rep(kern),
             'gate_warm':rep(lambda e:e.objects[0].bc.run(x,g)),
             'down_warm':rep(lambda e:e.objects[2].bc.run(h,y)),
             'adapter_cold':rot(lambda e:e.forward_batch(x)),
             'kern_cold':rot(kern),'only3_cold':rot(only3),
             'gate_cold':rot(lambda e:e.objects[0].bc.run(x,g)),
             'down_cold':rot(lambda e:e.objects[2].bc.run(h,y))}
        med,rng=bench(fns,blocks=150,repeats=1)
        sc=lambda k:NCOPY
        res={k:dict(med=med[k]/sc(k),min=rng[k][0]/sc(k),p10=rng[k][1]/sc(k)) for k in med}
        out[f'{bits}b_B{B}']=res;print(bits,B,' '.join(f"{k}={res[k]['med']:.1f}/{res[k]['p10']:.1f}/{res[k]['min']:.1f}" for k in res),flush=True)
json.dump(out,open('exl3_bench.json','w'),indent=1)
