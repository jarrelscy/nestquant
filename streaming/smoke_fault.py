"""C3 engine fault-injection smoke (one GPU, ~100 MB): N upgrades from a rank record file into a small device slot pool.
  NQ_FAULT_READ_MS=20 smoke_fault.py REPACK_DIR [N=40]  -> reads spaced >= 20 ms (slow drive): throughput ~ rec/20 ms
  NQ_FAULT_FAIL_PPM=500000 ...                           -> ~half the ops report failed; failed ops never post their row
Checks: every op completes (landed or failed), a failed op's seq is never written, a landed op's slot bytes == file bytes."""
import os,sys,time,torch,numpy as np
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import stream_engine as SE
rp=sys.argv[1];N=int(sys.argv[2]) if len(sys.argv)>2 else 40
rf=SE.RankFile(rp,0);rb=rf.rb;dev=torch.device('cuda',0)
slots=torch.zeros(N,rb,dtype=torch.uint8,device=dev);stage=torch.zeros(N,20,dtype=torch.int64,device=dev);seq=torch.zeros(N,dtype=torch.int32,device=dev)
eng=rf.engine(n_host=16,qd=8,device=0);rng=np.random.default_rng(0);recs=[int(x) for x in rng.choice(rf.idx['NE']*10,N,replace=False)]
t0=time.time()
for i,r in enumerate(recs):
    row=torch.full((20,),i+1,dtype=torch.int64)
    eng.upgrade(i,r,slots[i].data_ptr(),stage[i].data_ptr(),row,seq[i:].data_ptr(),i+1)
done={}
while len(done)<N and time.time()-t0<120:
    for tag,hit,trd,te2e in eng.poll():done[tag]=trd
    time.sleep(0.002)
dt=time.time()-t0;torch.cuda.synchronize();st=eng.stats();eng.close()
ok=[t for t,v in done.items() if v>=0];bad=[t for t,v in done.items() if v<0];sq=seq.cpu().numpy()
fd=os.open(rf.path,os.O_RDONLY);good=0
for t in ok[:8]:
    b=os.pread(fd,rb,recs[t]*rb);good+=bytes(slots[t].cpu().numpy().tobytes())==b
os.close(fd)
print(dict(env={k:v for k,v in os.environ.items() if k.startswith('NQ_FAULT')},n=N,completed=len(done),landed=len(ok),failed=len(bad),
           secs=round(dt,3),MBps=round(len(ok)*rb/dt/1e6,1),seq_written_on_failed=int(sum(sq[t]!=0 for t in bad)),
           seq_ok_on_landed=int(sum(sq[t]==t+1 for t in ok)),bytes_match=f'{good}/{min(8,len(ok))}',stats=st))
