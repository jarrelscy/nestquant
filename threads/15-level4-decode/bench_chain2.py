"""Full-expert A/B v2: additive progressive decoders (128-weight rings), unfused (5 launches) vs fused (2 launches), vs EXL3."""
import torch,json,sys,os;torch.cuda.set_per_process_memory_fraction(12/80)
import torch.nn.functional as F
from nq2 import *
import nq as v1
from timing import bench
from orbit_duet.exl3_adapter import EXL3Expert
NC=6
t2=json.load(open('tune2.json'));tf=json.load(open('tune_fused.json'));t1=json.load(open('tune.json'))
R='/home/coder/git/orbit-duet/runs/glm53_pilot_matched_l16/exl3_e36/'
CH=[(4,'T2H',2,{}),(2,'B2',2,{}),(3,'A3',2,{}),(3,'MIX',2,dict(grp=8)),(4,'A4',2,{}),(4,'A4',4,{}),(4,'T4',2,{})]
sg=lambda n:(torch.randint(0,2,(n,),device='cuda')*2-1).half()
sv_g,sv_u,su_d,su_x,sv_d=sg(2048),sg(2048),sg(2048),sg(6144),sg(6144)
out={}
for B in [1,2,3,4]:
    x32=torch.randn(B,6144,device='cuda')*0.01;xh=torch.empty(B,6144,device='cuda').half()
    acc=torch.zeros(B,4096,device='cuda');hh=torch.empty(B,2048,device='cuda').half()
    y=torch.zeros(B,6144,device='cuda');yo=torch.empty(B,6144,device='cuda').half()
    c1=torch.zeros(16,dtype=torch.int32,device='cuda');c2=torch.zeros(48,dtype=torch.int32,device='cuda')
    tb=1 if B<3 else 4
    accs=[torch.zeros(B,4096,device='cuda'),torch.zeros(B,4096,device='cuda')]
    fns={}
    for bits,d,G,kw in CH:
        gu=[Proj2(d,4096,6144,G=G,**kw) for _ in range(NC)];dn=[Proj2(d,6144,2048,G=G,**kw) for _ in range(NC)]
        cg=tuple(t2[f'{d}_G{G}|gu|{tb}']['cfg']);cd=tuple(t2[f'{d}_G{G}|down|{tb}']['cfg'])
        nm=f"{bits}b_{d}{'g8' if kw else ''}_G{G}"
        def unf(gu=gu,dn=dn,cg=cg,cd=cd):
            for i in range(NC):
                v1.M.had_in(x32,xh,su_x);gu[i](xh,acc,cg);v1.M.swiglu(acc,hh,sv_g,sv_u,su_d,y);dn[i](hh,y,cd);v1.M.had_in(y,yo,sv_d)
        def fus(gu=gu,dn=dn,cg=cg,cd=cd):
            for i in range(NC):
                gu[i](x32,acc,cg,mode=1,ex=(su_x,sv_g,sv_u,su_d,hh,c1));dn[i](hh,y,cd,mode=2,ex=(sv_o,yo,c2))
        fk=f"{d}{'g8' if kw else ''}_G{G}";fg_=tuple(tf[f'{fk}|gu|{tb}']['cfg']);fd_=tuple(tf[f'{fk}|down|{tb}']['cfg'])
        def fus2(gu=gu,dn=dn,cg=fg_,cd=fd_):
            for i in range(NC):
                a0,a1=accs[i%2],accs[(i+1)%2]
                gu[i](x32,a0,cg,mode=3,ex=(su_x,));dn[i](hh,y,cd,mode=4,ex=(a0,a1,sv_g,sv_u,su_d,sv_d,yo,c2))
        sv_o=sv_d
        fns[nm+'_unfused']=unf;fns[nm+'_fusedA']=fus;fns[nm+'_fusedB']=fus2
    # v1 reference: old T2R2 (64-weight rings, 3 HFMA) unfused
    gu=[v1.Proj('T2R2',4096,6144) for _ in range(NC)];dn=[v1.Proj('T2R2',6144,2048) for _ in range(NC)]
    cg=tuple(t1[f'T2R2|gu|{tb}']['cfg']);cd=tuple(t1[f'T2R2|down|{tb}']['cfg'])
    def old(gu=gu,dn=dn,cg=cg,cd=cd):
        for i in range(NC):
            v1.M.had_in(x32,xh,su_x);gu[i](xh,acc,cg);v1.M.swiglu(acc,hh,sv_g,sv_u,su_d,y);dn[i](hh,y,cd);v1.M.had_in(y,yo,sv_d)
    fns['4b_T2R2v1_unfused']=old
    gu=[v1.Proj('T2H',4096,6144) for _ in range(NC)];dn=[v1.Proj('T2H',6144,2048) for _ in range(NC)]
    cg=tuple(t1[f'T2H|gu|{tb}']['cfg']);cd=tuple(t1[f'T2H|down|{tb}']['cfg'])
    fns['4b_T2Hv1_unfused']=lambda gu=gu,dn=dn,cg=cg,cd=cd:old(gu,dn,cg,cd)
    for bits in [2,4]:
        ex=[EXL3Expert(R+f'expert_{bits}.bin') for _ in range(NC)]
        xhh=x32.half();g=torch.empty(B,2048,device='cuda');u=torch.empty(B,2048,device='cuda');h=torch.empty(B,2048,device='cuda',dtype=torch.half);yy=torch.empty(B,6144,device='cuda',dtype=torch.half)
        def kern(e):
            o=e.objects;o[0].bc.run(xhh,g);o[1].bc.run(xhh,u);torch.ops.aten.mul.out(F.silu(g),u,out=g);h.copy_(g);o[2].bc.run(h,yy)
        fns[f'{bits}b_EXL3_adapter']=(lambda ex=ex:[e.forward_batch(x32) for e in ex])
        fns[f'{bits}b_EXL3_kernels']=(lambda ex=ex,kern=kern:[kern(e) for e in ex])
    med,rng=bench(fns,blocks=int(os.environ.get('BLK',60)),repeats=1)
    out[B]={k:dict(med=v/NC,p10=rng[k][1]/NC) for k,v in med.items()}
    for k in sorted(out[B]):print(B,k,round(out[B][k]['med'],2),round(out[B][k]['p10'],2),flush=True)
    json.dump(out,open(os.environ.get('OUT','chain2.json'),'w'),indent=1)
    del fns;torch.cuda.empty_cache()
