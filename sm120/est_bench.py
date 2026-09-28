"""Microbenches behind the end-to-end tok/s estimate (TP4 shard: H=6144, I=512, 256 experts).
decode: NQ MoE layer us for c streams x t tokens/stream (t=4 = MTP ns=3 verify), level-4 pick share s drawn against
        a layer with 30% of experts at level 4 (the resident pool). B>8 runs as ceil(B/8) launches.
prefill: fp16 bmm expert GEMMs at W-token chunks (uniform W*8/256 tokens/expert), all-256-expert decode pass
        (4 launches of B=8 distinct = every expert streamed and decoded once), fp16 write bandwidth."""
import torch,random,json,itertools,sys,math
from moe import *;from routing import routing_tensors;from timing import bench,warmup
H,I,NP=6144,512,256;NPOOL=77;R=8
out=sys.argv[1] if len(sys.argv)>1 else 'results/est_bench.json'
ex=[Expert(H,I,seed=i) for i in range(NP)]
Ls={}
for name,lv in (('all2',lambda e:2),('all4',lambda e:4),('pool',lambda e:4 if e<NPOOL else 2)):
    L=MoELayer(NP,H,I,Bmax=8);[L.set(e,ex[e],lv(e)) for e in range(NP)];Ls[name]=L
def draw(rng,s,excl):
    src=range(NPOOL) if rng.random()<s else range(NPOOL,NP)
    c=[e for e in src if e not in excl];return rng.choice(c)
def sel_streams(c,t,s,rng):
    rows=[]
    for _ in range(c):
        prev=[]
        for j in range(t):
            row=[e for e in prev if rng.random()<0.4] if j else []
            while len(row)<8:row.append(draw(rng,s,row))
            rng.shuffle(row);rows.append(row);prev=row
    return rows
cgs=[[1,8,6],[1,8,3],[2,8,6],[1,8,12]];cds=[[1,8,4],[1,8,2],[2,8,2]]
share={1:0.656,2:0.63,4:0.60,8:0.56}
warmup();res={'decode':[],'prefill':{}}
for c,t in [(1,1),(1,4),(2,1),(2,4),(4,1),(4,4),(8,1),(8,4)]:
    B=c*t;rng=random.Random(100*c+t)
    fns={}
    for lname,s in (('all2',0.0),('all4',1.0),('pool',share[c])):
        rts=[]
        for i in range(R):
            rows=sel_streams(c,t,s if lname=='pool' else 0.5,rng)
            rts.append([routing_tensors(rows[k:k+8],seed=i) for k in range(0,B,8)])
        xs=[(torch.randn(min(8,B-k),H,device='cuda')*0.05).half() for k in range(0,B,8)]
        for cg,cd in itertools.product(cgs,cds):
            fns[(lname,tuple(cg),tuple(cd))]=(lambda L=Ls[lname],rts=rts,xs=xs,cg=cg,cd=cd:
                [L(x,sw[0],sw[1],cfg_gu=cg,cfg_dn=cd) for r in rts for x,sw in zip(xs,r)])
    med,_=bench(fns,blocks=10,repeats=1)
    for lname in ('all2','all4','pool'):
        k=min([k for k in med if k[0]==lname],key=lambda k:med[k])
        r=dict(c=c,t=t,B=B,layer=lname,share=share[c] if lname=='pool' else (0 if lname=='all2' else 1),us=round(med[k]/R,1),cfg=k[1:])
        res['decode'].append(r);print(r,flush=True)
# all-expert decode pass: 4 x B=8 distinct (64 experts each) = every expert once
for lname in ('all2','all4'):
    perm=list(range(NP));random.Random(7).shuffle(perm)
    rts=[routing_tensors([perm[k*64+j*8:k*64+j*8+8] for j in range(8)],seed=k) for k in range(4)]
    x=(torch.randn(8,H,device='cuda')*0.05).half()
    fns={(lname,tuple(cg),tuple(cd)):(lambda L=Ls[lname],cg=cg,cd=cd:[L(x,s_,w_,cfg_gu=cg,cfg_dn=cd) for s_,w_ in rts]) for cg,cd in itertools.product(cgs,cds)}
    med,_=bench(fns,blocks=10,repeats=1);k=min(med,key=med.get)
    res['prefill'][f'decode_all_{lname}_us']=round(med[k],1);print(lname,'all-expert decode pass us',round(med[k],1),k,flush=True)
del Ls,ex;torch.cuda.empty_cache()
# fp16 write bandwidth (materialising dequantised weights)
buf=torch.empty(2*1024**3,dtype=torch.half,device='cuda')
a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
for _ in range(3):buf.fill_(1.0)
a.record();[buf.fill_(1.0) for _ in range(10)];b.record();b.synchronize()
res['prefill']['write_GBps']=round(10*buf.numel()*2/(a.elapsed_time(b)/1e3)/1e9,1);print('write GB/s',res['prefill']['write_GBps'],flush=True)
del buf;torch.cuda.empty_cache()
# expert GEMMs per rank per layer: gate|up [E,T,H]@[E,H,2I], silu*mul, down [E,T,I]@[E,I,H]
wgu=(torch.randn(NP,H,2*I,device='cuda')*0.02).half();wd=(torch.randn(NP,I,H,device='cuda')*0.02).half()
for W in (4096,8192,16384,32768):
    T=W*8//NP
    x=(torch.randn(NP,T,H,device='cuda')*0.05).half()
    def f():
        g=torch.bmm(x,wgu);h=torch.nn.functional.silu(g[...,:I])*g[...,I:];return torch.bmm(h,wd)
    for _ in range(3):f()
    torch.cuda.synchronize();n=5
    a.record();[f() for _ in range(n)];b.record();b.synchronize();ms=a.elapsed_time(b)/n
    fl=2*NP*T*H*3*I
    res['prefill'][f'gemm_W{W}_ms']=round(ms,3);res['prefill'][f'gemm_W{W}_TFLOPS']=round(fl/ms/1e9,1)
    print('W',W,'expert GEMMs ms/layer',round(ms,3),'TFLOPS',round(fl/ms/1e9,1),flush=True)
    del x;torch.cuda.empty_cache()
json.dump(res,open(out,'w'),indent=1)
