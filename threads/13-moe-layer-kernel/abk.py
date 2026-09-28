"""A/B of NQ_RK_CODES build variants (interleaved graphs, one process, cold recency routing, tune2 per-B configs):
full layer us for level 2, 4 (K2 residual), 4p (gate|up 1.9375 / down 2.3125) and 4q (2 / 2.3125).
env I (2048 | 256), VARS ';'-separated variants: an NQ_RK_CODES value ('0xC1') or a ','-separated define list
('NQ_RK_GU=0x41,NQ_RK_DN=0x81', 'NQ_TAIL_BR') or 'def' (default build); levels run only where the build compiles their codes (mod.rk_codes)."""
import torch,os,json;torch.cuda.set_per_process_memory_fraction(12/80)
from moe import *;from common import routings;from timing import bench,warmup
from build import get
H,I=6144,int(os.environ.get('I',2048));NP=72 if I>=1024 else 256;R=8
VARS=os.environ.get('VARS','0x7;0xC7').split(';')
mods={v:get([] if v=='def' else ['NQ_RK_CODES='+v] if v.startswith('0x') else v.split(',')) for v in VARS}
tune=json.load(open(f'tune2_I{I}.json'))['default']
LV={2:(2,0,0),4:(4,0,0),'4p':(4,6,7),'4q':(4,0,7)}
exs={}
for k,(lv,rg,rd) in LV.items():
    key=(rg,rd)
    if key not in exs:exs[key]=[Expert(H,I,seed=3000+i,rk_gu=rg,rk_dn=rd,var=True) for i in range(NP)]
Ls={}
for v in VARS:
    cg_,cd_=mods[v].rk_codes()
    for k,(lv,rg,rd) in LV.items():
        if not (cg_>>rg&1 and cd_>>rd&1):continue
        L=MoELayer(NP,H,I,mod=mods[v]);[L.set(e,exs[(rg,rd)][e],lv) for e in range(NP)];Ls[v,k]=L
warmup();out=[]
for B in [1,2,3,4]:
    rts=routings(B,NP,R);x=(torch.randn(B,H,device='cuda')*0.05).half()
    cg=tune[f'gu|B{B}']['cfg4'];cd=tune[f'dn|B{B}']['cfg4'];cg2=tune[f'gu|B{B}']['cfg2'];cd2=tune[f'dn|B{B}']['cfg2']
    fns={(v,k):(lambda L=L,k=k:[L(x,s,w,cfg_gu=cg2 if k==2 else cg,cfg_dn=cd2 if k==2 else cd) for s,w in rts]) for (v,k),L in Ls.items()}
    med,_=bench(fns,blocks=12,repeats=1)
    row={f'{v}|{k}':round(med[(v,k)]/R,1) for (v,k) in Ls};print('B',B,row,flush=True);out.append(dict(B=B,**row))
json.dump(out,open(f'/tmp/nestquant/13-moe-layer-kernel/abk_I{I}.json','w'),indent=1)
