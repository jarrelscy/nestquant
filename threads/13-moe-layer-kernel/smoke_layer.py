"""Real-layer smoke: one uploaded nestquant-v1 layer (all 256 experts, lr plane, default allocation) through nqmoe.cu
vs the sum over routed experts of rw * moe.Expert.ref (same fp16 rounding points), gate max rel err <= 1.5e-4.
Allocations: the manifest default (level4_experts at L4, rest L2), all L4, all L2. Routings: random top-8 B1-B4, plus a
sweep that routes every expert (B4 x 8 slots, permutation), with random softmax-like fp16 weights; x = 0.05 randn,
also with 4 massive-activation channels (x 60). Configs: tune2_I2048 cfg2/cfg4 per B, persistent [2,8,4,1], grid [1,8,3]/[1,8,2].
The kernel-plane tensors stay on GPU; the ref-only unpacked words (p4w, Mb, Nn) stay on CPU and move per call.
Also (B1, default allocation) the kernel vs thread 12's nq_decode.decode_expert dense forward (fp32, no fp16 act rounding).
usage: smoke_layer.py [LAYER_DIR=/tmp/nestquant/nq-encode-v1/L38]"""
import sys,json,time,collections,torch;torch.cuda.set_per_process_memory_fraction(12/80)
from verify_t12 import load_expert,scales,D
from moe import *
dev='cuda';root=sys.argv[1] if len(sys.argv)>1 else '/tmp/nestquant/nq-encode-v1/L38'
man=json.load(open(f'{root}/manifest.json'));E=man['n_experts'];L4=set(man['default_allocation']['level4_experts'])
rk=collections.Counter(tuple(man['per_expert'][str(e)]['lr_rank'].get(p,0) for p in ('gate','up','down')) for e in range(E))
print(f'layer {man["layer"]}: {E} experts, {len(L4)} at L4 by default; lr ranks gate/up/down: {dict(rk.most_common())}',flush=True)
REF=('p4w','Mb','Nn');exs=[];sg={};t=time.time()
for e in range(E):
    ex=load_expert(f'{root}/experts/E{e}.pt',verbose=False);ex.Qg=ex.Qu=ex.Qd=None
    for p in (ex.gu,ex.dn):
        for k in REF:setattr(p,k,getattr(p,k).cpu())
    sg[e]={L:scales(ex.art,L) for L in (2,4)};ex.art=None;exs.append(ex)
    if e%64==63:print(f'  loaded {e+1} ({time.time()-t:.0f} s, gpu {torch.cuda.memory_allocated()/1e9:.2f} GB)',flush=True)
H,I=exs[0].H,exs[0].I;Lk=MoELayer(E,H,I,G=4)
tune=json.load(open('tune2_I2048.json'))['default']
def setlv(lv):
    for e,ex in enumerate(exs):ex.signs=sg[e][lv[e]];Lk.set(e,ex,lv[e])
def ref(x,sel,rw,lv):
    y=torch.zeros(x.shape[0],H,device=dev);xf=x.float()
    for b in range(x.shape[0]):
        for k in range(sel.shape[1]):
            e=int(sel[b,k]);ex=exs[e];ex.signs=sg[e][lv[e]]
            for p in (ex.gu,ex.dn):
                for q in REF:setattr(p,q,getattr(p,q).to(dev))
            y[b]+=rw[b,k].float()*Expert.ref(ex,xf[b:b+1],lv[e])[0]
            for p in (ex.gu,ex.dn):
                for q in REF:setattr(p,q,getattr(p,q).cpu())
    return y
rel=lambda a,b:((a-b).norm()/b.norm()).item()
g=torch.Generator(device='cpu').manual_seed(0)
def weights(B):
    w=torch.softmax(torch.randn(B,8,generator=g)*1.5,1);return w.half().to(dev)
def xs(B,spike):
    x=torch.randn(B,H,generator=g)*0.05
    if spike:x[:,[17,1234,3000,6000]]*=60
    return x.half().to(dev)
ALLOC={'default':[4 if e in L4 else 2 for e in range(E)],'allL4':[4]*E,'allL2':[2]*E}
worst=0;ok=True;n=0;seen=set()
for an,lv in ALLOC.items():
    setlv(lv);cases=[]
    for B in (1,2,3,4):
        for i in range(3):cases.append((B,torch.stack([torch.randperm(E,generator=g)[:8] for _ in range(B)]).to(dev)))
    perm=torch.randperm(E,generator=g).view(-1,4,8)   # every expert once
    cases+=[(4,p.to(dev)) for p in perm]
    wa=0
    for ci,(B,sel) in enumerate(cases):
        rw=weights(B);x=xs(B,ci%2==1);yr=ref(x,sel,rw,lv)
        cf=[(tune[f'gu|B{B}']['cfg2' if an=='allL2' else 'cfg4'],tune[f'dn|B{B}']['cfg2' if an=='allL2' else 'cfg4']),([2,8,4,1],[2,8,4,1])]
        if ci<12:cf.append(([1,8,3],[1,8,2]))
        for cg,cd in cf:
            y=Lk(x,sel,rw,cfg_gu=cg,cfg_dn=cd).clone();r=rel(y,yr);wa=max(wa,r);n+=1
            if not torch.isfinite(y).all() or r>1.5e-4:ok=False;print(f'  FAIL {an} case {ci} B{B} cfg {cg}/{cd}: rel {r:.2e}',flush=True)
        seen|=set(sel.flatten().tolist())
    clean=Lk.zws[:32*(I//128)*4].abs().max().item()==0 and Lk.wq.abs().sum().item()==0 and Lk.acc_gu.abs().max().item()==0 and Lk.acc_d.abs().max().item()==0
    ok&=clean;worst=max(worst,wa)
    print(f'{an}: {len(cases)} batches (B1-4 random x 3 each + {len(perm)} B4 batches routing all {E} experts), max rel err vs Expert.ref {wa:.2e}, workspace clean {clean}',flush=True)
# kernel vs nq_decode dense (B1, default allocation): fp16 activation rounding floor
setlv(ALLOC['default']);lv=ALLOC['default'];ds=[]
for i in range(4):
    sel=torch.randperm(E,generator=g)[:8].view(1,8).to(dev);rw=weights(1);x=xs(1,i%2==1)
    y=Lk(x,sel,rw).clone();yd=torch.zeros(1,H,device=dev)
    for k in range(8):
        e=int(sel[0,k]);art=torch.load(f'{root}/experts/E{e}.pt',weights_only=False,map_location='cpu')
        Wg,Wu,Wd=D.decode_expert(art,lv[e],dev);xf=x.float()
        yd+=rw[0,k].float()*((torch.nn.functional.silu(xf@Wg.T)*(xf@Wu.T))@Wd.T);del Wg,Wu,Wd
    ds.append(rel(y,yd))
print(f'kernel vs nq_decode.decode_expert dense (B1 x 4, default allocation, 8 experts each): {min(ds):.2e} .. {max(ds):.2e}')
print(f'{n} launches, {len(seen)} distinct experts routed, max rel err vs Expert.ref {worst:.2e} (gate 1.5e-4), peak gpu {torch.cuda.max_memory_allocated()/1e9:.2f} GB')
print('PASS' if ok else 'FAIL')
