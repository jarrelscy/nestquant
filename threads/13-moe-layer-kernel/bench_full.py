"""Paired benchmark, cold L2: NestQuant grouped MoE layer vs EXL3 1.5.1 exl3_moe_coop.
B1-4 x {2b, 4b, 50/50 mix} x {recency, distinct}. Each graph replays R=8 layer calls whose routings avoid the previous
call's experts (72-expert pool, 1.4-3 GB of weights >> 40 MB L2). us per layer and effective GB/s (bytes of distinct
experts touched / time)."""
import torch,random,json,sys,os;torch.cuda.set_per_process_memory_fraction(12/80)
from moe import *;from exl3_moe import *;from routing import *;from timing import bench,warmup
H,I=6144,2048;NP=72;R=8
tune=json.load(open('tune_I2048.json'))
ex=[Expert(H,I,seed=i) for i in range(NP)]
nq_bytes={2:ex[0].bytes(2),4:ex[0].bytes(4)}
lv_of={'2b':lambda e:2,'4b':lambda e:4,'mix':lambda e:2 if e<NP//2 else 4}
Ls={}
for k,f in lv_of.items():
    L=MoELayer(NP,H,I);[L.set(e,ex[e],f(e)) for e in range(NP)];Ls[k]=L
G2=EXL3Group(NP,2,seed=1);G4=EXL3Group(NP,4,seed=2);Gm2=EXL3Group(NP//2,2,seed=3);Gm4=EXL3Group(NP//2,4,seed=4)
ex3_bytes={2:G2.bytes_per_expert,4:G4.bytes_per_expert}
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
        ms={'2b':EXL3MoE([(G2,0)],B),'4b':EXL3MoE([(G4,0)],B),'mix':EXL3MoE([(Gm2,0),(Gm4,NP//2)],B)}
        fns={}
        for k in lv_of:
            fns[('exl3',k)]=(lambda m=ms[k]:[m(x,s,w) for s,w in rts])
            fns[('nq',k)]=(lambda L=Ls[k]:[L(x,s,w,cfg_gu=cg,cfg_dn=cd) for s,w in rts])
        med,_=bench(fns,blocks=30,repeats=1)
        for k,f in lv_of.items():
            dist=[set(e for r in rows for e in r) for rows in rows_all]
            nd=sum(len(d) for d in dist)/R
            by_nq=sum(sum(nq_bytes[f(e)] for e in d) for d in dist)/R
            by_ex=sum(sum(ex3_bytes[f(e)] for e in d) for d in dist)/R
            tn,te=med[('nq',k)]/R,med[('exl3',k)]/R
            r=dict(mode=mode,B=B,level=k,distinct=nd,nq_us=tn,exl3_us=te,nq_GBs=by_nq/tn/1e3,exl3_GBs=by_ex/te/1e3,speedup=te/tn)
            res.append(r);print(json.dumps({a:(round(b,2) if isinstance(b,float) else b) for a,b in r.items()}),flush=True)
json.dump(res,open('bench_full.json','w'),indent=1)
