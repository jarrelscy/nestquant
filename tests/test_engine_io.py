"""nqstream engine nq-io extensions, on a real rank file (GPU, small: ~100 MB device + CUDA context).
  python tests/test_engine_io.py REPACK_DIR ALT_DIR [rank]
Checks byte-exact device slots + stage row + seq for: one drive (original path), dual drive (both drives used),
RAM tier (tier records served from the tier, no SSD read), cancel (ops not started are dropped and never written),
tier_drop (tier freed, later upgrades read the SSD again)."""
import os,sys,time,random,torch,numpy as np
HERE=os.path.dirname(os.path.abspath(__file__));sys.path.insert(0,HERE+'/../streaming')
import stream_engine as SE
rp,alt=sys.argv[1],sys.argv[2];rank=int(sys.argv[3]) if len(sys.argv)>3 else 3
rf=SE.RankFile(rp,rank);rb=rf.rb;N=rf.idx['NE']*len(rf.idx['layers']);dev=torch.device('cuda',0)
NS=48;slots=torch.empty(NS,rb,dtype=torch.uint8,device=dev);stage=torch.zeros(NS,20,dtype=torch.int64,device=dev);seq=torch.zeros(NS,dtype=torch.int32,device=dev)
fd=os.open(rf.path,os.O_RDONLY);rng=random.Random(1);ok=True
def ref(r):return np.frombuffer(os.pread(fd,rb,r*rb),np.uint8)
def run(eng,recs,cancel=(),tag0=0):
    slots.fill_(0xAB);stage.zero_();seq.zero_();torch.cuda.synchronize()
    for i,r in enumerate(recs):
        row=torch.full((20,),1000+i,dtype=torch.int64)
        eng.upgrade(tag0+i,r,slots[i].data_ptr(),stage[i].data_ptr(),row,seq.data_ptr()+4*i,7+i)
    for i in cancel:eng.cancel(tag0+i)
    got={};t=time.time()
    while len(got)<len(recs) and time.time()-t<30:
        for tg,hit,trd,te in eng.poll():got[tg-tag0]=(hit,trd)
        time.sleep(0.001)
    torch.cuda.synchronize();assert len(got)==len(recs),f'{len(got)}/{len(recs)} ops completed'
    nc=0;sl=slots.cpu().numpy();st=stage.cpu().numpy();sq=seq.cpu().numpy()
    for i,r in enumerate(recs):
        hit,trd=got[i]
        if trd<=-1e8:
            nc+=1;assert i in cancel and (sl[i]==0xAB).all() and st[i].sum()==0 and sq[i]==0,f'cancelled op {i} wrote the device'
        else:
            assert trd>=0,f'op {i} failed {trd}';assert (sl[i]==ref(r)).all(),f'slot {i} rec {r} bytes differ'
            assert (st[i]==1000+i).all() and sq[i]==7+i,f'op {i} row/seq'
    return got,nc
def chk(name,cond,msg=''):
    global ok;ok&=bool(cond);print(f'{"PASS" if cond else "FAIL"} {name} {msg}',flush=True)
recs=rng.sample(range(N),NS)
e=rf.engine(64,8,0);g,_=run(e,recs);s=e.stats();chk('one drive',s['drives']==1 and s['bytes_read']==NS*rb,f"drive_reads {s['drive_reads']}");e.close()
e=rf.engine(64,4,0,alt_path=f'{alt}/rank{rank}.bin',qd_alt=4);g,_=run(e,rng.sample(range(N),NS));s=e.stats()
chk('dual drive',s['drives']==2 and min(s['drive_reads'])>0 and sum(s['drive_reads'])==NS,f"drive_reads {s['drive_reads']} read_s {[round(x,3) for x in s['drive_read_s']]}");e.close()
tier=rng.sample(range(N),32);e=rf.engine(64,8,0,alt_path=f'{alt}/rank{rank}.bin');n=e.tier_load(tier)
mix=tier[:24]+rng.sample(sorted(set(range(N))-set(tier)),24);rng.shuffle(mix);g,_=run(e,mix,tag0=100);s=e.stats()
chk('RAM tier',n==32 and s['tier_hits']==24 and s['bytes_read']==24*rb and s['tier_bytes']==24*rb,f"tier_hits {s['tier_hits']} ssd reads {sum(s['drive_reads'])}")
e.tier_drop();t=time.time()
while e.stats()['tier_state']!=3 and time.time()-t<5:time.sleep(0.01)
g,_=run(e,tier[:16],tag0=200);s2=e.stats()
chk('tier_drop',s2['tier_state']==3 and s2['tier_hits']==24 and s2['bytes_read']-s['bytes_read']+rb*(s2['host_hits']-s['host_hits'])==16*rb,
    f"state {s2['tier_state']} tier_hits {s2['tier_hits']} new ssd bytes {(s2['bytes_read']-s['bytes_read'])//rb} recs, lru hits {s2['host_hits']-s['host_hits']}");e.close()
e=rf.engine(1,1,0);rc=rng.sample(range(N),20);g,nc=run(e,rc,cancel=set(range(5,20)),tag0=300);s=e.stats()
chk('cancel',nc>=10 and s['cancelled']==nc,f'{nc}/15 cancelled before start, rest completed byte-exact');e.close()
print('PASS' if ok else 'FAIL');sys.exit(0 if ok else 1)
