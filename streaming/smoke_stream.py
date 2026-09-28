"""End-to-end upgrade path on the production format: repacked record file -> O_DIRECT read into pinned bounce ->
H2D into a slot (side stream) -> mailbox post -> captured graph [mailbox.apply, MoE layer] replay.
  smoke_stream.py ROOT REPACK L [rank=0] [tp=4] [nslot=16] [steps=200] [engine=0: python pread | 1: nqstream io_uring engine, n_host=max(4,nslot/2), qd 4]
Checks: (1) each slot's bytes equal the resident kernel tensors (format); (2) every replay's output equals a
reference layer whose table holds the levels that replay applied, using resident level-4 planes (rel err ~0);
(3) slot reuse after downgrade (slot scribbled with junk first) never leaks into the output; (4) the table row after each
applied op equals the posted row. Pass: every step rel err <= max(1e-4, 2 x reference-vs-itself) (the kernel is not bitwise repeatable: reference-vs-itself
reaches ~1e-4 from K-split atomic order through the fp16 h) and no wrong table rows."""
import os,sys,json,time,random,mmap,torch,numpy as np
HERE=os.path.dirname(os.path.abspath(__file__));sys.path[:0]=[HERE,HERE+'/../sm120']
torch.cuda.set_per_process_memory_fraction(12/96)
import nqload as NQ,p4rec as PR;from moe import MoELayer,Mailbox,entry
root,rp,L=sys.argv[1],sys.argv[2],int(sys.argv[3]);a=[int(x) for x in sys.argv[4:]]+[None]*5
rank,tp,NSLOT,NSTEP,ENG=a[0] or 0,a[1] or 4,a[2] or 16,a[3] or 200,a[4] or 0
dev='cuda';idx=json.load(open(f'{rp}/rank{rank}.json'));rb=idx['rec_bytes'];lay=dict(seg=idx['seg'],rec_bytes=rb)
assert str(L) in idx['layers'],f'L{L} not repacked'
RL=NQ.RankLayer(root,L,rank,tp);ids=RL.experts;NE_=max(ids)+1;H,I=RL.H,RL.I
for ex in RL.ex.values():ex.signs=ex.sc[2]
print(f'L{L} rank{rank}/TP{tp}: {len(ids)} experts, record {rb} B, slots {NSLOT}',flush=True)
# pinned, page-aligned bounce (cudaHostAlloc) for O_DIRECT; one record per upgrade
fd=os.open(f'{rp}/rank{rank}.bin',os.O_RDONLY|os.O_DIRECT)
bounce=torch.empty(NSLOT*rb,dtype=torch.uint8).pin_memory();assert bounce.data_ptr()%4096==0
bv=memoryview(bounce.numpy())
slots=torch.empty(NSLOT,rb,dtype=torch.uint8,device=dev)
junk=torch.randint(0,256,(rb,),dtype=torch.uint8).pin_memory()
def read(E,sl):
    t=time.perf_counter();n=os.preadv(fd,[bv[sl*rb:(sl+1)*rb]],((L-idx['L0'])*idx['NE']+E)*rb);assert n==rb,n
    return time.perf_counter()-t
# (1) format: record bytes == resident tensors
bad=0
for E in ids[:min(len(ids),32)]:
    read(E,0);rec=bounce[:rb];ex=RL.ex[E]
    for k,t in PR._tensors(ex).items():
        o,_=lay['seg'][k];raw=t.contiguous().cpu().view(torch.uint8).view(-1)
        if not torch.equal(rec[o:o+raw.numel()],raw):bad+=1;print('  mismatch',E,k)
print(f'  record bytes vs resident: {"OK" if bad==0 else f"{bad} MISMATCH"} ({min(len(ids),32)} experts)',flush=True)
# (2)/(3) streamed layer vs reference layer
M=MoELayer(NE_,H,I,Bmax=4);Ref=MoELayer(NE_,H,I,Bmax=4);MB=Mailbox(M)
def setlv(Lr,E,lv):ex=RL.ex[E];ex.signs=ex.sc[lv];Lr.set(E,ex,lv);ex.signs=ex.sc[2]
for E in ids:setlv(M,E,2);setlv(Ref,E,2)
B=4;TK=min(8,len(ids));rng=random.Random(L*10+rank)
sel=torch.tensor([rng.sample(ids,TK) for _ in range(B)],device=dev,dtype=torch.long)
rw=torch.softmax(torch.randn(B,TK,device=dev),-1).half()   # the kernel reads rw as fp16
x=(torch.randn(B,H,device=dev)*0.05).half()
s=torch.cuda.Stream();side=torch.cuda.Stream()
if ENG:
    import stream_engine as SE;eng=SE.mod().Engine(f'{rp}/rank{rank}.bin',rb,max(4,NSLOT//2),4,torch.cuda.current_device());elat=[];ehit=0;kind={};ntag=0
f=lambda:(MB.apply(),M(x,sel,rw))
with torch.cuda.stream(s):
    for _ in range(3):f()
torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
with torch.cuda.graph(g):f()
torch.cuda.synchronize()
lev={E:2 for E in ids};pend={};slot_of={};free=list(range(NSLOT));worst=0;lat=[];napplied=0;floor=0;nover=0;want={};ntbl=0;nbig=0;sel_set=sorted(set(sel.flatten().tolist()))
for st in range(NSTEP):
    for _ in range(rng.randint(0,3)):
        idle=[E for E in ids if E not in pend];upc=[E for E in idle if lev[E]==2 and E not in slot_of];dn=[E for E in idle if lev[E]==4]
        if free and upc and (not dn or rng.random()<0.6):
            E=rng.choice([e for e in upc if e in sel_set] or upc) if rng.random()<0.7 else rng.choice(upc)
            sl=free.pop();slot_of[E]=sl;rw_=PR.row(RL.ex[E],lay,slots[sl].data_ptr(),entry);want[E]=rw_;pend[E]=4
            if ENG:ntag+=1;kind[ntag]=4;MB.hseq[E]+=1;eng.upgrade(ntag,(L-idx['L0'])*idx['NE']+E,slots[sl].data_ptr(),MB.stage[E].data_ptr(),rw_,MB.seq.data_ptr()+4*E,MB.hseq[E])
            else:
                lat.append(read(E,sl))
                with torch.cuda.stream(side):slots[sl].copy_(bounce[sl*rb:(sl+1)*rb],non_blocking=True)
                MB.post(E,rw_,side)
        elif dn:
            E=rng.choice(dn);rw_=entry(RL.ex[E],2);want[E]=rw_;pend[E]=2
            if ENG:ntag+=1;kind[ntag]=2;MB.hseq[E]+=1;eng.post(ntag,MB.stage[E].data_ptr(),rw_,MB.seq.data_ptr()+4*E,MB.hseq[E])
            else:MB.post(E,rw_,side)
    g.replay();torch.cuda.current_stream().synchronize();o=M.out[:B].float().clone()
    if ENG:
        for tag,hit,trd,te2e in eng.poll():
            assert trd>=0,f'read failed res {trd}'
            if kind.pop(tag)==4:ehit+=hit;elat.append(te2e);(lat.append(trd) if not hit else None)
    for E in list(pend):
        if MB.done(E):
            lev[E]=pend.pop(E);napplied+=1
            if lev[E]==2:
                sl=slot_of.pop(E)
                with torch.cuda.stream(side):slots[sl].copy_(junk,non_blocking=True)
                side.synchronize();free.append(sl)
    # the replay applied every op done() now reports (done is read after the replay)
    tb=M.table.cpu()
    for E in want:
        if E not in pend and not torch.equal(tb[E],want[E]):ntbl+=1;print(f'    step {st}: table row of E{E} != posted row (level {int(tb[E][0])})')
        if E in pend and torch.equal(tb[E],want[E]):print(f'    step {st}: E{E} row applied but done() false')
    for E in sel_set:setlv(Ref,E,lev[E])
    y=Ref(x,sel,rw).float().clone();y2=Ref(x,sel,rw).float()
    e=((o-y).norm()/y.norm()).item();fl=((y2-y).norm()/y.norm()).item();floor=max(floor,fl);worst=max(worst,e);nover+=e>max(1e-4,2*fl)
    if e>1e-5:nbig+=1
    if e>1e-4:print(f'    step {st}: rel {e:.2e} (ref repeat {fl:.2e}) levels sel {[lev[E] for E in sel_set]}')
side.synchronize();torch.cuda.synchronize()
if ENG:
    import time as _t;_t.sleep(0.05);[elat.append(t4) for tg,_,_,t4 in eng.poll() if kind.pop(tg)==4];st_=eng.stats();eng.close()
    el=np.array(elat)*1e3;print(f'  engine: {st_["upgrades"]} upgrades ({ehit} host-LRU hits), {st_["posts"]} posts; op end-to-end p50 {np.percentile(el,50):.3f} p99 {np.percentile(el,99):.3f} ms')
lat=np.array(lat)*1e3
ok=bad==0 and nover==0 and ntbl==0;print(f'  table rows wrong after apply: {ntbl}')
print(f'  {NSTEP} replays, {napplied} ops applied, worst rel err vs reference {worst:.2e} (reference repeat {floor:.2e}, {nbig} steps >1e-5, {nover} over the bound); O_DIRECT read p50 {np.percentile(lat,50):.3f} '
      f'p99 {np.percentile(lat,99):.3f} ms ({len(lat)} reads, {rb/np.median(lat)/1e6:.2f} GB/s at qd 1)')
print('STREAM SMOKE','PASS' if ok else 'FAIL')
