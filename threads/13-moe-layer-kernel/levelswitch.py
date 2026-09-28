"""Level switching under one captured CUDA graph.
Upgrade 2->4: side-stream pinned H2D of P4+d4 into a preallocated device slot, event, main waits, table row flip.
Downgrade 4->2: main flips table row first, event, side stream may then overwrite / reuse the slot.
All steps are enqueued asynchronously (no host sync between steps) and every replay's output is checked
against a dense reference for the level state the replay must have seen."""
import torch,random,json;torch.cuda.set_per_process_memory_fraction(12/80)
from moe import *;from routing import make_sel,routing_tensors
H,I=6144,2048;NP=16;B=4;NSLOT=4;dev='cuda'
ex=[Expert(H,I,seed=100+i) for i in range(NP)]
L=MoELayer(NP,H,I);L.cfg_gu=[2,8,4];L.cfg_dn=[2,8,4]
# slot layout: [gu_p4 | gu_d4 | dn_p4 | dn_d4], each 64 KiB aligned (int32 words)
al=lambda n:(n+16383)//16384*16384
zg,zd=ex[0].gu.z,ex[0].dn.z
offs=[0];[offs.append(offs[-1]+al(n)) for n in (zg['p4'],zg['d4'],zd['p4'],zd['d4'])]
SW=offs[-1];print('slot bytes',SW*4)
slots=torch.randint(-2**31,2**31-1,(NSLOT,SW),dtype=torch.int32,device=dev)   # garbage
host={}                                                                           # pinned P4 images per expert
for e in range(NP):
    h=torch.empty(SW,dtype=torch.int32).pin_memory()
    for o,t in zip(offs,(ex[e].gu.p4,ex[e].gu.d4,ex[e].dn.p4,ex[e].dn.d4)):h[o:o+t.numel()]=t.cpu()
    host[e]=h
junk=torch.randint(-2**31,2**31-1,(SW,),dtype=torch.int32).pin_memory()
def row(e,level,slot):
    r=entry(ex[e],2);r[0]=level
    if slot is not None:
        b=slots[slot].data_ptr();r[2],r[3],r[6],r[7]=b+4*offs[0],b+4*offs[1],b+4*offs[2],b+4*offs[3]
    return r
for e in range(NP):L.set(e,ex[e],2)
# routing: all 16 experts get used across the 4 tokens (recency-like overlap)
sel,rw=routing_tensors(make_sel(B,list(range(NP)),'recency',rng=random.Random(3)),seed=3)
x=(torch.randn(B,H,device=dev)*0.05).half()
print('sel',sel.tolist())
# per-(expert, level) reference contributions
xf=x.float();C={(e,l):ex[e].ref(xf,l) for e in range(NP) for l in (2,4)}
def ref(levels):
    y=torch.zeros(B,H,device=dev)
    for b in range(B):
        for k in range(8):e=int(sel[b,k]);y[b]+=float(rw[b,k])*C[(e,levels[e])][b]
    return y
# capture once
main=torch.cuda.current_stream();side=torch.cuda.Stream()
hits=torch.zeros(NP,dtype=torch.int32).pin_memory();L.hits_ptr=hits.data_ptr()   # host-mapped (UVA) routing-hit export
s=torch.cuda.Stream()
with torch.cuda.stream(s):
    for _ in range(3):L(x,sel,rw)
torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
with torch.cuda.graph(g):L(x,sel,rw)
torch.cuda.synchronize();hits.zero_()
# random async schedule
rng=random.Random(0);lev=[2]*NP;slot_of={};free=list(range(NSLOT));steps=[];outs=[];pins=[]
NSTEP=200
for st in range(NSTEP):
    ops=[]
    for _ in range(rng.randint(1,3)):
        up=[e for e in range(NP) if lev[e]==2];dn=[e for e in range(NP) if lev[e]==4]
        if free and up and (not dn or rng.random()<0.5):
            e=rng.choice(up);sl=free.pop()
            with torch.cuda.stream(side):slots[sl].copy_(host[e],non_blocking=True);ev=torch.cuda.Event();ev.record(side)
            main.wait_event(ev)
            r=row(e,4,sl).pin_memory();pins.append(r);L.table[e].copy_(r,non_blocking=True)
            lev[e]=4;slot_of[e]=sl;ops.append(('up',e,sl))
        elif dn:
            e=rng.choice(dn);sl=slot_of.pop(e)
            r=row(e,2,None).pin_memory();pins.append(r);L.table[e].copy_(r,non_blocking=True)
            ev=torch.cuda.Event();ev.record(main);side.wait_event(ev)
            with torch.cuda.stream(side):slots[sl].copy_(junk,non_blocking=True)   # scribble the freed slot
            free.append(sl);lev[e]=2;ops.append(('down',e,sl))
    g.replay();outs.append(L.out[:B].clone());steps.append((list(lev),ops))
torch.cuda.synchronize()
errs=[];wrong=[];prev=[2]*NP;routed=set(sel.flatten().tolist())
for (lv,ops),o in zip(steps,outs):
    y=ref(lv);errs.append(((o-y).norm()/y.norm()).item())
    # distance to the stale (pre-step) state, where a routed expert changed level: shows the check has teeth
    if any(prev[e]!=lv[e] for e in routed):alt=ref(prev);wrong.append(((o-alt).norm()/alt.norm()).item())
    prev=lv
nup=sum(1 for _,ops in steps for op in ops if op[0]=='up');ndn=sum(1 for _,ops in steps for op in ops if op[0]=='down')
exp_hits=torch.bincount(sel.flatten().cpu(),minlength=NP)*NSTEP
res=dict(hits_export_exact=bool(torch.equal(hits.long(),exp_hits)),steps=NSTEP,upgrades=nup,downgrades=ndn,max_rel_err=max(errs),mean_rel_err=sum(errs)/len(errs),
         min_rel_err_vs_wrong_state=min(wrong),max_l4_at_once=max(sum(l==4 for l in lv) for lv,_ in steps),slot_bytes=SW*4)
print(json.dumps(res,indent=1));json.dump(res,open('levelswitch.json','w'),indent=1)
