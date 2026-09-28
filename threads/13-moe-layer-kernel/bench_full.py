"""Paired benchmark, cold L2: NestQuant grouped MoE layer vs EXL3 1.5.1 exl3_moe_coop.
B1-4 x {2b, 4b, 50/50 mix, 4f = level 4 with fractional residual gu K=1.75 / dn K=2.5 (3.75/4.5 bpw),
4p = T14 production pattern gu 1.9375 / dn 2.3125 (4.0846 bpw), 4q = gu 2 / dn 2.3125 (4.1263); 4p/4q with base-variant
sign planes as T12 production} x {recency, distinct}; env LEVELS (comma list, default 2b,4b,mix,4p,4q; 4f needs NQ_DEFS=NQ_RK_CODES=0x7), OUT (json). Each graph replays R=8 layer calls whose routings avoid the previous
call's experts (72-expert pool, 1.4-3 GB of weights >> 40 MB L2). us per layer and effective GB/s (bytes of distinct
experts touched / time)."""
import torch,random,json,sys,os;torch.cuda.set_per_process_memory_fraction(12/80)
from moe import *;from exl3_moe import *;from routing import *;from timing import bench,warmup
H,I=6144,2048;NP=72;R=8
tune=json.load(open('tune2_I2048.json'))['default']
LEV=os.environ.get('LEVELS','2b,4b,mix,4p,4q').split(',')
FR={'4f':(1,2,False),'4p':(6,7,True),'4q':(0,7,True)}          # level-4 key -> (gu code, dn code, base-variant signs)
ex=[Expert(H,I,seed=i) for i in range(NP)]
exs={k:[Expert(H,I,seed=1000*(j+1)+i,rk_gu=FR[k][0],rk_dn=FR[k][1],var=FR[k][2]) for i in range(NP)] for j,k in enumerate(FR) if k in LEV}
nq_bytes={2:ex[0].bytes(2),4:ex[0].bytes(4),**{k:v[0].bytes(4) for k,v in exs.items()}}
lv_of={k:f for k,f in {'2b':lambda e:2,'4b':lambda e:4,'mix':lambda e:2 if e<NP//2 else 4,'4f':lambda e:4,'4p':lambda e:4,'4q':lambda e:4}.items() if k in LEV}
Ls={}
for k,f in lv_of.items():
    L=MoELayer(NP,H,I);[L.set(e,exs.get(k,ex)[e],f(e)) for e in range(NP)];Ls[k]=L
G2=EXL3Group(NP,2,seed=1);G4=EXL3Group(NP,4,seed=2);Gm2=EXL3Group(NP//2,2,seed=3);Gm4=EXL3Group(NP//2,4,seed=4)
ex3_bytes={2:G2.bytes_per_expert,4:G4.bytes_per_expert,**{k:G4.bytes_per_expert for k in FR}}
print('bytes/expert nq',nq_bytes,'exl3',ex3_bytes,flush=True)
res=[]
warmup()
for mode in ['recency','distinct']:
    for B in [1,2,3,4]:
        rng=random.Random(B);rts=[];prev=();rows_all=[]
        for i in range(R):
            rows=make_sel(B,range(NP),mode,rng=rng,avoid=prev);prev=[e for r in rows for e in r];rts.append(routing_tensors(rows,seed=i));rows_all.append(rows)
        x=(torch.randn(B,H,device='cuda')*0.05).half()
        cg,cd=tune[f'gu|B{B}']['cfg'],tune[f'dn|B{B}']['cfg']
        ms={'2b':EXL3MoE([(G2,0)],B),'4b':EXL3MoE([(G4,0)],B),'mix':EXL3MoE([(Gm2,0),(Gm4,NP//2)],B)};ms.update({k:ms['4b'] for k in FR})
        fns={}
        for k in lv_of:
            if k not in FR:fns[('exl3',k)]=(lambda m=ms[k]:[m(x,s,w) for s,w in rts])
            fns[('nq',k)]=(lambda L=Ls[k]:[L(x,s,w,cfg_gu=cg,cfg_dn=cd) for s,w in rts])
        med,_=bench(fns,blocks=30,repeats=1)
        for k,f in lv_of.items():
            dist=[set(e for r in rows for e in r) for rows in rows_all]
            nd=sum(len(d) for d in dist)/R
            by_nq=sum(sum(nq_bytes[k if k in FR else f(e)] for e in d) for d in dist)/R
            by_ex=sum(sum(ex3_bytes[f(e)] for e in d) for d in dist)/R
            tn,te=med[('nq',k)]/R,med[('exl3','4b' if k in FR else k)]/R
            r=dict(mode=mode,B=B,level=k,distinct=nd,nq_us=tn,exl3_us=te,nq_GBs=by_nq/tn/1e3,exl3_GBs=by_ex/te/1e3,speedup=te/tn)
            res.append(r);print(json.dumps({a:(round(b,2) if isinstance(b,float) else b) for a,b in r.items()}),flush=True)
json.dump(res,open(os.environ.get('OUT','bench_full.json'),'w'),indent=1)
