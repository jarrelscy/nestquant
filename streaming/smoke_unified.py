"""nqstream direct mode (NQ_UNIFIED, unified memory / DGX Spark) on a synthetic record file (GPU, small).
  python smoke_unified.py [DIR=/tmp/nestquant/36-spark-land/ue_test] [nrec=128] [rec_bytes=2854912]
Checks, for the original path (device slots + bounce) and the direct path (Engine.alloc_slots host-mapped slots, reads
straight into the slot): byte-exact slots, stage row + seq landed, cancel of not-started ops writes nothing, a failed
read (past EOF) posts nothing, a GPU kernel reads the mapped slots correctly (device copy + per-row sums), and no
bounce / host hits in direct mode. Prints read throughput of both paths (zero-copy GPU reads on a discrete GPU go over
PCIe, so only the engine side is comparable here)."""
import os,sys,time,random,torch,numpy as np
HERE=os.path.dirname(os.path.abspath(__file__));sys.path.insert(0,HERE)
import stream_engine as SE
D=sys.argv[1] if len(sys.argv)>1 else '/tmp/nestquant/36-spark-land/ue_test'
NR=int(sys.argv[2]) if len(sys.argv)>2 else 128;rb=int(sys.argv[3]) if len(sys.argv)>3 else 2854912
os.makedirs(D,exist_ok=True);path=f'{D}/rank0.bin'
if not os.path.exists(path) or os.path.getsize(path)!=NR*rb:
    g=np.random.default_rng(7)
    with open(path,'wb') as f:
        for i in range(NR):f.write(g.integers(0,256,rb,dtype=np.uint8).tobytes())
fd=os.open(path,os.O_RDONLY)
def ref(r):return np.frombuffer(os.pread(fd,rb,r*rb),np.uint8)
dev=torch.device('cuda',0);M=SE.mod();ok=True;NS=48
stage=torch.zeros(NS,20,dtype=torch.int64,device=dev);seq=torch.zeros(NS,dtype=torch.int32,device=dev)
def chk(name,cond,msg=''):
    global ok;ok&=bool(cond);print(f'{"PASS" if cond else "FAIL"} {name} {msg}',flush=True)
def run(eng,slots,recs,cancel=(),tag0=0):
    slots.fill_(0xAB);stage.zero_();seq.zero_();torch.cuda.synchronize()
    t=time.time()
    for i,r in enumerate(recs):
        eng.upgrade(tag0+i,r,slots[i].data_ptr(),stage[i].data_ptr(),torch.full((20,),1000+i,dtype=torch.int64),seq.data_ptr()+4*i,7+i)
    for i in cancel:eng.cancel(tag0+i)
    got={}
    while len(got)<len(recs) and time.time()-t<60:
        for tg,hit,trd,te in eng.poll():got[tg-tag0]=(hit,trd)
        time.sleep(0.0005)
    dt=time.time()-t;torch.cuda.synchronize();assert len(got)==len(recs),f'{len(got)}/{len(recs)} ops completed'
    sl=slots.cpu().numpy();st=stage.cpu().numpy();sq=seq.cpu().numpy();nc=nf=0;bad=[]
    for i,r in enumerate(recs):
        hit,trd=got[i]
        if trd<=-1e8:
            nc+=1
            if not (i in cancel and (sl[i]==0xAB).all() and st[i].sum()==0 and sq[i]==0):bad.append(('cancel',i))
        elif trd<0:
            nf+=1
            if not (st[i].sum()==0 and sq[i]==0):bad.append(('failed posted',i))
        elif not ((sl[i]==ref(r)).all() and (st[i]==1000+i).all() and sq[i]==7+i):bad.append(('bytes/row/seq',i,r))
    return dict(nc=nc,nf=nf,bad=bad,dt=dt)
recs=random.Random(1).sample(range(NR),NS)
for direct in (False,True):
    nm='direct' if direct else 'bounce'
    e=M.Engine(path,rb,64,8,0,'',0,direct)
    slots=e.alloc_slots(NS) if direct else torch.empty(NS,rb,dtype=torch.uint8,device=dev)
    r=run(e,slots,recs);s=e.stats()
    chk(f'{nm}: {NS} upgrades byte-exact + row/seq',not r['bad'] and r['nf']==0,f"bad {r['bad'][:3]} {NS*rb/r['dt']/1e9:.2f} GB/s engine")
    chk(f'{nm}: stats',s['direct']==direct and s['bytes_read']==NS*rb and s['host_hits']==0,f"bytes {s['bytes_read']//rb} recs, host_hits {s['host_hits']}")
    # GPU kernels on the slot memory (mapped host memory in direct mode)
    cp=slots.clone();sums=slots.view(NS,-1).to(torch.int64).sum(1).cpu().numpy()
    cs=np.array([int(ref(rc).astype(np.int64).sum()) for rc in recs])
    chk(f'{nm}: GPU reads slots (copy + row sums)',torch.equal(cp.cpu(),slots.cpu()) and (sums==cs).all())
    del cp
    # same records again: bounce path serves host LRU hits, direct path re-reads the SSD
    r=run(e,slots,recs,tag0=1000);s2=e.stats()
    chk(f'{nm}: repeat',not r['bad'],f"host_hits {s2['host_hits']-s['host_hits']}, ssd recs {(s2['bytes_read']-s['bytes_read'])//rb}")
    if direct:chk('direct: no host hits',s2['host_hits']==0 and s2['bytes_read']==2*NS*rb)
    # failed reads: record index past EOF
    r=run(e,slots,[NR+5,NR+9]+recs[:6],tag0=2000)
    chk(f'{nm}: failed reads post nothing',r['nf']==2 and not r['bad'],f"failed {r['nf']} bad {r['bad'][:3]}")
    if direct:
        try:e.tier_load([0]);chk('direct: tier refused',False)
        except RuntimeError:chk('direct: tier refused',True)
    e.close()
    # cancel: qd 1, most ops still queued
    e=M.Engine(path,rb,1,1,0,'',0,direct);sl2=e.alloc_slots(NS) if direct else slots
    rc=random.Random(3).sample(range(NR),20);r=run(e,sl2,rc,cancel=set(range(5,20)),tag0=3000);s=e.stats()
    chk(f'{nm}: cancel',r['nc']>=10 and s['cancelled']==r['nc'] and not r['bad'],f"{r['nc']}/15 cancelled, rest byte-exact")
    e.close();del slots,sl2
# slot tensors outlive the engine (deleter independent of it)
e=M.Engine(path,rb,64,8,0,'',0,True);sl=e.alloc_slots(4);e.close();sl.fill_(3);torch.cuda.synchronize()
chk('alloc_slots outlives engine',int(sl.sum())==3*4*rb);del sl
torch.cuda.synchronize();os.close(fd)
print('PASS' if ok else 'FAIL');sys.exit(0 if ok else 1)
