"""pg53 stage 1 probe (experimental, default off; NQ_PGPROBE=1 at boot): per-layer decode GPU timestamps at MoE start /
end of every NQ layer call with <= 8 tokens, plus an end-to-end "a load triggered at layer L-1 lands before MoE(L)"
measurement with real record reads (csrc/nq_pgprobe.cu, host thread without the GIL).
Control file /dev/shm/nq_pgprobe (re-read every 0.5 s): "run pt n stride"
   run 0/1, pt = trigger point on layer L-1 (0 = MoE start of L-1 = h after L-1 attention, source b; 1 = MoE end of
   L-1 ~ h entering L before the MoE all-reduce, source a upper bound), n = records read per trigger (0 = ack only),
   stride: only target layers L with L % stride == 0.
Dump: write a tag into /dev/shm/nq_pgprobe_dump -> /dbg/pgprobe_<tag>_r<rank>.npz (hdr, ts ring, hlog, calib)."""
import os,threading,time,numpy as np,torch
from torch.utils.cpp_extension import load
NLM,NS,NR,NF=96,4096,8192,4
OFF_TRIG=16;OFF_ACK=OFF_TRIG+NR*4;OFF_TS=OFF_ACK+NLM;OFF_HLOG=OFF_TS+NS*NLM*NF
CTL='/dev/shm/nq_pgprobe';DUMP='/dev/shm/nq_pgprobe_dump'

class Probe:
    def __init__(s,rank,first,log):
        load(name='nq_pgprobe',sources=[os.path.join(os.path.dirname(os.path.abspath(__file__)),'csrc','nq_pgprobe.cu')],
             extra_include_paths=['/data/Jarrel/liburing/include'],extra_ldflags=['/data/Jarrel/liburing/lib/liburing.a'],
             extra_cuda_cflags=['-O3'],is_python_module=False,verbose=False)
        P=torch.ops.nq_pgprobe;s.P=P;s.rank=rank;s.first=first;s.log=log
        s.buf=torch.zeros(int(P.total_words()),dtype=torch.int64,pin_memory=True);s.b=s.buf.numpy();s.ptr=s.buf.data_ptr()
        s.cal=torch.zeros(1,dtype=torch.int64,pin_memory=True);s.off,s.rtt=P.calib(s.cal.data_ptr())
        alt=os.environ.get('NQ_IO_SPLIT_RANKS','2,3').split(',')
        path=f"/nqrepack1/rank{rank}.bin" if str(rank) in alt else f"/nqrepack/rank{rank}.bin"
        import json;rb=json.load(open(f"/nqrepack/rank{rank}.json"))['rec_bytes'];nrec=os.path.getsize(path)//rb
        s.scr=torch.empty(8*rb,dtype=torch.uint8,device='cuda')
        P.start(s.ptr,path,rb,nrec,s.scr.data_ptr(),8,torch.cuda.current_device())
        s.m=None;threading.Thread(target=s._ctl,daemon=True,name='nq-pgprobe').start()
        log.info('NestQuant pgprobe rank %d: on (%s, rb %d, %d recs), gpu-host offset %d ns (rtt %d ns)',rank,path,rb,nrec,s.off,s.rtt)
    def stamp(s,L,pt,T):s.P.stamp(s.ptr,L,pt,T,int(L==s.first))
    def _ctl(s):
        while True:
            time.sleep(0.5)
            try:
                m=os.stat(CTL).st_mtime_ns
                if m!=s.m:
                    s.m=m;f=[int(x) for x in open(CTL).read().split()]
                    s.b[2],s.b[3],s.b[4]=f[1],f[2],f[3];s.b[5]=f[0]
                    s.log.info('NestQuant pgprobe rank %d: ctl run %d pt %d n %d stride %d',s.rank,*f[:4])
            except OSError:pass
            except Exception:s.log.exception('pgprobe ctl')
            try:
                if os.path.exists(DUMP):
                    tag=open(DUMP).read().strip()
                    out=f'/dbg/pgprobe_{tag}_r{s.rank}.npz'
                    if tag and not os.path.exists(out):
                        b=s.b.copy()
                        np.savez(out,hdr=b[:16],ts=b[OFF_TS:OFF_TS+NS*NLM*NF].reshape(NS,NLM,NF),
                                 hlog=b[OFF_HLOG:OFF_HLOG+NR*6].reshape(NR,6),off=s.off,rtt=s.rtt)
                        s.log.info('NestQuant pgprobe rank %d: dumped %s',s.rank,out)
            except Exception:s.log.exception('pgprobe dump')
