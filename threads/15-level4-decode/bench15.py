"""Full-expert chain A/B (one process, CUDA graphs, shuffled replay, cold L2 via NC=6 copies): thread-15 variants (fused-B 2 launches
and unfused 5 launches) vs thread-04 A4 (nqk2) vs EXL3 adapter / bare kernels, B = 1..4."""
import torch,json,sys,os;torch.cuda.set_per_process_memory_fraction(12/80)
import torch.nn.functional as F
from nq15 import Proj,M
import nq2
from timing import bench
from orbit_duet.exl3_adapter import EXL3Expert
NC=6
t15=json.load(open('tune15.json'));t04=json.load(open('tune_fused.json'))
R='/home/coder/git/orbit-duet/runs/glm53_pilot_matched_l16/exl3_e36/'
CH=[('2b_B2',0,0),('2b_B2greedy',34,34),('2b_B2raw',40,40),
    ('4b_A4',3,3),('4b_A4funnel',4,4),('4b_foldf32',6,6),('4b_P',33,33),('4b_Praw',41,41),('4b_P2',37,37),('4b_P2raw',42,42),
    ('frac_P_r1.75gu_r2.5dn',29,30),('frac_Praw_r1.75gu_r2.5dn',43,44),('frac_P2raw_r1.75gu_r2.5dn',46,45)]
UNF=['4b_A4','4b_P','4b_P2']
if os.environ.get('VARS'):CH=[c for c in CH if c[0] in os.environ['VARS'].split(',')];UNF=[u for u in UNF if u in os.environ['VARS'].split(',')]
sg=lambda n:(torch.randint(0,2,(n,),device='cuda')*2-1).half()
sv_g,sv_u,su_d,su_x,sv_d=sg(2048),sg(2048),sg(2048),sg(6144),sg(6144)
out={}
Bs=[int(b) for b in os.environ.get('BS','1,2,3,4').split(',')]
for B in Bs:
    x32=torch.randn(B,6144,device='cuda')*0.01;xh=torch.empty(B,6144,device='cuda').half()
    acc=torch.zeros(B,4096,device='cuda');hh=torch.empty(B,2048,device='cuda').half()
    y=torch.zeros(B,6144,device='cuda');yo=torch.empty(B,6144,device='cuda').half()
    c2=torch.zeros(48,dtype=torch.int32,device='cuda')
    tb=1 if B<3 else 4
    accs=[torch.zeros(B,4096,device='cuda'),torch.zeros(B,4096,device='cuda')]
    fns={}
    for nm,vg,vd in CH:
        gu=[Proj(vg,4096,6144,seed=i) for i in range(NC)];dn=[Proj(vd,6144,2048,seed=10+i) for i in range(NC)]
        cg=tuple(t15[f'{vg}|gu|{tb}']['cfg']);cd=tuple(t15[f'{vd}|down|{tb}']['cfg'])
        def fus(gu=gu,dn=dn,cg=cg,cd=cd):
            for i in range(NC):
                a0,a1=accs[i%2],accs[(i+1)%2]
                gu[i](x32,a0,cg,mode=3,ex=(su_x,));dn[i](hh,y,cd,mode=4,ex=(a0,a1,sv_g,sv_u,su_d,sv_d,yo,c2))
        fns[nm+'_fusedB']=fus
        if nm in UNF:
            def unf(gu=gu,dn=dn,cg=cg,cd=cd):
                for i in range(NC):
                    M.had_in(x32,xh,su_x);gu[i](xh,acc,cg);M.swiglu(acc,hh,sv_g,sv_u,su_d,y);dn[i](hh,y,cd);M.had_in(y,yo,sv_d)
            fns[nm+'_unfused']=unf
    gu=[nq2.Proj2('A4',4096,6144,G=2) for _ in range(NC)];dn=[nq2.Proj2('A4',6144,2048,G=2) for _ in range(NC)]
    fg_=tuple(t04[f'A4_G2|gu|{tb}']['cfg']);fd_=tuple(t04[f'A4_G2|down|{tb}']['cfg'])
    def t04f(gu=gu,dn=dn,cg=fg_,cd=fd_):
        for i in range(NC):
            a0,a1=accs[i%2],accs[(i+1)%2]
            gu[i](x32,a0,cg,mode=3,ex=(su_x,));dn[i](hh,y,cd,mode=4,ex=(a0,a1,sv_g,sv_u,su_d,sv_d,yo,c2))
    fns['4b_A4_thread04_fusedB']=t04f
    for bits in [2,4]:
        ex=[EXL3Expert(R+f'expert_{bits}.bin') for _ in range(NC)]
        xhh=x32.half();g=torch.empty(B,2048,device='cuda');u=torch.empty(B,2048,device='cuda');h=torch.empty(B,2048,device='cuda',dtype=torch.half);yy=torch.empty(B,6144,device='cuda',dtype=torch.half)
        def kern(e,g=g,u=u,h=h,yy=yy,xhh=xhh):
            o=e.objects;o[0].bc.run(xhh,g);o[1].bc.run(xhh,u);torch.ops.aten.mul.out(F.silu(g),u,out=g);h.copy_(g);o[2].bc.run(h,yy)
        fns[f'{bits}b_EXL3_adapter']=(lambda ex=ex:[e.forward_batch(x32) for e in ex])
        fns[f'{bits}b_EXL3_kernels']=(lambda ex=ex,kern=kern:[kern(e) for e in ex])
    med,rng=bench(fns,blocks=int(os.environ.get('BLK',60)),repeats=1)
    out[B]={k:dict(med=v/NC,p10=rng[k][1]/NC) for k,v in med.items()}
    for k in sorted(out[B]):print(B,k,round(out[B][k]['med'],2),round(out[B][k]['p10'],2),flush=True)
    json.dump(out,open(os.environ.get('OUT','chain15.json'),'w'),indent=1)
    del fns;torch.cuda.empty_cache()
