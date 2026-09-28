"""Resident-file round trip: experts loaded from OUT/res/rank{r}/L{L}.pt (+ records from rank{r}.bin at level 4) must
give the same table rows' outputs as nqload.RankLayer built from the fit shards.
  check_resident.py ROOT REPACK L [rank=0] [tp=4] [trials=4]
Compares the two MoE layers on random tokens at level 2 (all experts) and level 4 (a random quarter, P4 from the
record file for the resident copy, resident P4 for the RankLayer copy). Pass: rel err <= max(1e-4, 2 x repeat noise)."""
import os,sys,json,random,torch
HERE=os.path.dirname(os.path.abspath(__file__));sys.path[:0]=[HERE,HERE+'/../sm120']
torch.cuda.set_per_process_memory_fraction(40/96)
import nqload as NQ,resident as RS,stream_engine as SE,p4rec as PR
from moe import MoELayer,entry
root,rp,L=sys.argv[1],sys.argv[2],int(sys.argv[3]);a=[int(x) for x in sys.argv[4:]]+[None]*3;rank=a[0] or 0;tp=a[1] or 4;NT=a[2] or 4
dev='cuda';NE=256
RL=NQ.RankLayer(root,L,rank,tp);ex,H,I=RS.load(f'{rp}/res/rank{rank}/L{L}.pt',dev);assert (H,I)==(RL.H,RL.I) and sorted(ex)==sorted(RL.ex)
rf=SE.RankFile(rp,rank);rb=rf.rb;rng=random.Random(L*7+rank)
up=sorted(rng.sample(sorted(ex),NE//4));slots=torch.empty(len(up),rb,dtype=torch.uint8,device=dev)
fd=os.open(rf.path,os.O_RDONLY)
for i,E in enumerate(up):slots[i].copy_(torch.frombuffer(bytearray(os.pread(fd,rb,rf.rec(L,E)*rb)),dtype=torch.uint8))
os.close(fd)
A=MoELayer(NE,H,I,Bmax=8);B=MoELayer(NE,H,I,Bmax=8)
for E in RL.experts:
    x=RL.ex[E];x.signs=x.sc[2];A.set(E,x,2);B.table[E].copy_(entry(ex[E],2))
bad=0;worst=0
for lv in (2,4):
    if lv==4:
        for i,E in enumerate(up):
            x=RL.ex[E];x.signs=x.sc[4];A.set(E,x,4);x.signs=x.sc[2]
            B.table[E].copy_(PR.row(ex[E],rf.lay,slots[i].data_ptr(),entry))
    for t in range(NT):
        Bt=8;sel=torch.stack([torch.tensor(rng.sample(up if lv==4 and t%2 else sorted(ex),8)) for _ in range(Bt)]).to(dev)
        rw=torch.softmax(torch.randn(Bt,8,device=dev),1).half();xx=(torch.randn(Bt,H,device=dev)*0.05).half()
        ya=A(xx,sel,rw).float().clone();ya2=A(xx,sel,rw).float().clone();yb=B(xx,sel,rw).float().clone()
        er=((yb-ya).norm()/ya.norm()).item();fl=((ya2-ya).norm()/ya.norm()).item();worst=max(worst,er)
        ok=er<=max(1e-4,2*fl);bad+=not ok
        print(f'  level {lv} trial {t}: rel {er:.2e} (noise {fl:.2e}) {"ok" if ok else "BAD"}',flush=True)
print(json.dumps(dict(L=L,rank=rank,worst=float(f'{worst:.3e}'),bad=bad)));print('RESIDENT CHECK','PASS' if bad==0 else 'FAIL')
