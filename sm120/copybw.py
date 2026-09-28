"""576 KiB pinned H2D chunk bandwidth, idle vs while the MoE graph replays; MoE slowdown from the copies."""
import torch,json,time;torch.cuda.set_per_process_memory_fraction(12/80)
from moe import *;from common import *;from routing import *;from timing import warmup
H,I=6144,2048;NP=72;R=8;CH=576*1024;NCH=64
ex=[Expert(H,I,seed=i) for i in range(NP)]
tune=json.load(open('tune_I2048.json'))
host=torch.empty(NCH,CH,dtype=torch.uint8).pin_memory();host.random_(0,255)
devb=torch.empty(NCH,CH,dtype=torch.uint8,device='cuda')
side=torch.cuda.Stream();main=torch.cuda.current_stream()
def copies(rounds):
    a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(side):
        a.record(side)
        for _ in range(rounds):
            for i in range(NCH):devb[i].copy_(host[i],non_blocking=True)   # one cudaMemcpyAsync per 576 KiB chunk
        b.record(side)
    return a,b
res={}
warmup()
for lvname,lvf in [('2b',lambda e:2),('4b',lambda e:4)]:
    L=MoELayer(NP,H,I);[L.set(e,ex[e],lvf(e)) for e in range(NP)]
    for B in [1,4]:
        L.cfg_gu=tune[f'gu|B{B}']['cfg'];L.cfg_dn=tune[f'dn|B{B}']['cfg']
        rts=routings(B,NP,R);x=(torch.randn(B,H,device='cuda')*0.05).half()
        f=lambda:[L(x,s,w) for s,w in rts]
        s=torch.cuda.Stream()
        with torch.cuda.stream(s):
            for _ in range(3):f()
        torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):f()
        def moe(n):
            a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
            a.record();[g.replay() for _ in range(n)];b.record();return a,b
        N=300
        torch.cuda.synchronize();a,b=moe(N);b.synchronize();t_alone=a.elapsed_time(b)*1e3/N/R
        torch.cuda.synchronize();ca,cb=copies(8);cb.synchronize();bw_idle=8*NCH*CH/(ca.elapsed_time(cb)*1e-3)/1e9
        # concurrent: start copies, then MoE replays; copy volume sized to overlap the MoE window
        torch.cuda.synchronize()
        rounds=max(2,int(N*R*t_alone*1e-6*bw_idle*1e9*0.8/(NCH*CH)))
        ca,cb=copies(rounds);a,b=moe(N);torch.cuda.synchronize()
        tc=ca.elapsed_time(cb)*1e-3;tm=a.elapsed_time(b)*1e-3
        bw_conc=rounds*NCH*CH/tc/1e9;t_conc=tm*1e6/N/R
        k=f'{lvname} B{B}';res[k]=dict(moe_us_alone=round(t_alone,1),moe_us_during_copy=round(t_conc,1),
            slowdown_pct=round(100*(t_conc/t_alone-1),1),h2d_GBps_idle=round(bw_idle,2),h2d_GBps_during_moe=round(bw_conc,2),
            copy_s=round(tc,3),moe_s=round(tm,3),us_per_576KiB_during=round(CH/(bw_conc*1e9)*1e6,1))
        print(k,res[k],flush=True)
    del L;torch.cuda.empty_cache()
json.dump(res,open('copybw.json','w'),indent=1)
