"""Level switching with a captured mailbox kernel: the host never touches the compute stream between replays.
Graph = [mailbox.apply, MoE layer]. Scheduler ops go only to a side stream; the replay is issued immediately (racing
the side stream). The state each replay saw is read back from applied (host-mapped) and checked vs the dense ref."""
import torch,random,json;torch.cuda.set_per_process_memory_fraction(12/80)
from moe import *;from routing import make_sel,routing_tensors
H,I=6144,2048;NP=16;B=4;NSLOT=4;dev='cuda'
ex=[Expert(H,I,seed=100+i) for i in range(NP)]
L=MoELayer(NP,H,I);L.cfg_gu=[2,8,4];L.cfg_dn=[2,8,4];MB=Mailbox(L)
al=lambda n:(n+16383)//16384*16384
zg,zd=ex[0].gu.z,ex[0].dn.z
offs=[0];[offs.append(offs[-1]+al(n)) for n in (zg['p4'],zg['d4'],zd['p4'],zd['d4'])];SW=offs[-1]
slots=torch.randint(-2**31,2**31-1,(NSLOT,SW),dtype=torch.int32,device=dev)
host={}
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
sel,rw=routing_tensors(make_sel(B,list(range(NP)),'recency',rng=random.Random(3)),seed=3)
x=(torch.randn(B,H,device=dev)*0.05).half();xf=x.float()
C={(e,l):ex[e].ref(xf,l) for e in range(NP) for l in (2,4)}
def ref(levels):
    y=torch.zeros(B,H,device=dev)
    for b in range(B):
        for k in range(8):e=int(sel[b,k]);y[b]+=float(rw[b,k])*C[(e,levels[e])][b]
    return y
side=torch.cuda.Stream();s=torch.cuda.Stream()
f=lambda:(MB.apply(),L(x,sel,rw))
with torch.cuda.stream(s):
    for _ in range(3):f()
torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
with torch.cuda.graph(g):f()
torch.cuda.synchronize()
rng=random.Random(1);lev=[2]*NP;pend={};slot_of={};free=list(range(NSLOT));errs=[];stale=[];NSTEP=300;lag=[]
for st in range(NSTEP):
    prev=list(lev)
    # scheduler: 0-2 new ops on the side stream, no host sync, experts with an outstanding op are skipped
    for _ in range(rng.randint(0,2)):
        idle=[e for e in range(NP) if e not in pend]
        up=[e for e in idle if lev[e]==2 and e not in slot_of];dn=[e for e in idle if lev[e]==4]
        if free and up and (not dn or rng.random()<0.5):
            e=rng.choice(up);sl=free.pop();slot_of[e]=sl
            with torch.cuda.stream(side):slots[sl].copy_(host[e],non_blocking=True)
            MB.post(e,row(e,4,sl),side);pend[e]=(4,st)
        elif dn:
            e=rng.choice(dn);MB.post(e,row(e,2,None),side);pend[e]=(2,st)
    g.replay();torch.cuda.current_stream().synchronize()      # (sync only to check this replay's output)
    o=L.out[:B].clone()
    for e in list(pend):                                      # which ops did this replay's mailbox apply?
        if MB.done(e):
            lvn,st0=pend.pop(e);lev[e]=lvn;lag.append(st-st0)
            if lvn==2:                                        # released: scribble + recycle the slot immediately
                sl=slot_of.pop(e)
                with torch.cuda.stream(side):slots[sl].copy_(junk,non_blocking=True)
                free.append(sl)
    y=ref(lev);errs.append(((o-y).norm()/y.norm()).item())
    if lev!=prev and any(lev[e]!=prev[e] for e in set(sel.flatten().tolist())):
        a=ref(prev);stale.append(((o-a).norm()/a.norm()).item())
torch.cuda.synchronize()
res=dict(steps=NSTEP,ops=len(lag),applied_same_replay=sum(1 for l in lag if l==0),applied_next=sum(1 for l in lag if l==1),
         max_lag=max(lag),max_rel_err=max(errs),min_rel_err_vs_stale=min(stale))
print(json.dumps(res,indent=1));json.dump(res,open('levelswitch_mbox.json','w'),indent=1)
