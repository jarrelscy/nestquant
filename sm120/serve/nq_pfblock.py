# nq-kld: NQ_PF_BLOCK_MS -- block each prefill MoE layer on its scheduled level-4 planes (default off, eval knob).
# During a prefill chunk (the lookahead path of nq_vllm.forward), before layer L's mailbox apply, every rank waits until
#   * rank 0: the lookahead plan for (L, this chunk) exists (made while servicing layer L-1's router stats) and
#   * followers: rank 0 has published that plan's oplog position (shm marker) and this rank's log reader is past it, and
#   * this rank's executor has no SSD/tier read in flight for layer L (X.ops), i.e. the planned planes have landed
#     (their mailbox rows are then picked up by the apply that follows),
# or until the per-layer budget runs out. The wait is a condition wait on the streaming thread's iteration cv (it
# notifies at the end of every iteration; we wake it so it polls the engine sooner) -- no host spin.
# Budget: env NQ_PF_BLOCK_MS (0 = off, <0 = unlimited (safety cap NQ_PF_BLOCK_CAP_S, 20 s)), overridden in-boot by the
# file /dev/shm/nq_pf_block (same units), re-read at the first MoE layer of each chunk on every rank.
# Per chunk each rank logs: layers waited / timed out, total and max wait ms.
import os,time,numpy as np,torch
try:
    from vllm.logger import init_logger;log=init_logger('vllm.nestquant.pfblock')
except Exception:
    import logging;log=logging.getLogger('nestquant')
XB=65536
KNOB=os.environ.get('NQ_PF_BLOCK_KNOB','/dev/shm/nq_pf_block');CAP_S=float(os.environ.get('NQ_PF_BLOCK_CAP_S','20'))

class PFBlock:
    def __init__(s,rt,L_):
        s.rt=rt;s.L_=list(L_);s.li={L:i for i,L in enumerate(s.L_)};s.env=float(os.environ.get('NQ_PF_BLOCK_MS','0'))
        s.ms=s.env;s.km=None;s.cnt=0;s.mk=None;s.st=None
        s.denv=float(os.environ.get('NQ_DEC_BLOCK_MS','0'));s.dms=s.denv;s.dkm=None;s.dseq=0;s.dm=None;s.dst=dict(n=0,to=0,tot=0.,max=0.)
        key=getattr(rt,'iokey',None);s.path=f'{key}_pfblk.bin' if key else None
        if s.path and rt.rank==0:
            s.mk=np.memmap(s.path,dtype=np.int64,mode='w+',shape=(len(s.L_),3));s.mk[:]=0;s.mk.flush()
        s.dpath=f'{key}_decblk.bin' if key else None
        if s.dpath and rt.rank==0:
            s.dm=np.memmap(s.dpath,dtype=np.int64,mode='w+',shape=(3,));s.dm[:]=0;s.dm.flush()
        log.info('NestQuant pf-block rank %d: installed (budget %s ms, knob %s, marker %s)',rt.rank,s.env,KNOB,s.path)
    # ---- leader (streaming thread, LA.service): the plan for (Lp, tc) is issued; its ups are in the log up to here
    def publish(s,Lp,tc,oplog):
        if s.mk is None or Lp not in s.li:return
        i=s.li[Lp]
        if oplog is not None:s.mk[i,1]=oplog.gen;s.mk[i,2]=oplog.off
        s.mk[i,0]=tc
    def _marker(s):
        if s.mk is None and s.path and os.path.exists(s.path):
            try:s.mk=np.memmap(s.path,dtype=np.int64,mode='r',shape=(len(s.L_),3))
            except Exception:s.mk=None
        return s.mk
    def _knob(s):
        try:m=os.stat(KNOB).st_mtime_ns
        except OSError:s.ms=s.env;s.km=None;return
        if m!=s.km:
            try:s.ms=float(open(KNOB).read().split()[0]);s.km=m
            except Exception:s.ms=s.env
    def _ready(s,L,c):
        rt=s.rt;X=rt.X;i=s.li[L]
        if not(L==s.L_[0] and c==1):                 # first layer of the very first chunk: nothing was planned for it
            m=s._marker()
            if m is None or m[i,0]<c:return False      # rank 0 publishes the plan for (L, chunk c) after its oplog put
            if rt.F is not None:
                lg=rt.log
                if (lg.gen,lg.off)<(int(m[i,1]),int(m[i,2])):return False     # this rank has not replayed it yet
                # (no check of the follower's coalescing queue: an entry held there behind a landed-but-unapplied op of
                #  layer L only clears after this layer's mailbox apply, i.e. after this wait -> it would deadlock)
        for v in X.ops.values():
            if v[0]==L and v[2]==4:return False
        return True
    def wait(s,L):
        if L==s.L_[0]:s.cnt+=1;s._knob();s.st=dict(n=0,to=0,tot=0.,max=0.,Lmax=-1)
        if s.ms==0 or s.st is None:return
        rt=s.rt;c=s.cnt;t0=time.perf_counter();dl=t0+(s.ms/1e3 if s.ms>0 else CAP_S);ok=False
        if rt.F is None and rt.LA is not None and L!=s.L_[0] and rt.LA.cid!=c and not getattr(s,'warned',0):
            s.warned=1;log.warning('NestQuant pf-block: chunk counter %d != lookahead cid %d',c,rt.LA.cid)
        with rt.cv:
            while True:
                if not rt.in_iter and s._ready(L,c):ok=True;break
                r=dl-time.perf_counter()
                if r<=0:break
                rt.wake.set();rt.cv.wait(min(r,0.002))
        dt=(time.perf_counter()-t0)*1e3;st=s.st;st['n']+=1;st['tot']+=dt;st['to']+=0 if ok else 1
        if dt>st['max']:st['max']=dt;st['Lmax']=L
        if L==s.L_[-1]:
            log.info('NestQuant pf-block rank %d chunk %d: budget %s ms, waited %d layers, timed out %d, total %.1f ms, max %.1f ms (L%d)',
                     rt.rank,c,s.ms,st['n'],st['to'],st['tot'],st['max'],st['Lmax'])

    # ---- NQ_DEC_BLOCK_MS: decode steps. Decode runs in CUDA graphs (the MoE forward is replayed, no host code per layer),
    # so the wait is per STEP, before the step's graph launch: after the previous step finished (sync), rank 0 lets its
    # streaming loop run 2 full iterations (it has seen that step's hits and issued the resulting upgrades), publishes its
    # oplog position; followers wait until they replayed up to it; then every rank waits until it has no upgrade read in
    # flight (any layer). Budget per step = NQ_DEC_BLOCK_MS x number of MoE layers (<0: unlimited, cap CAP_S).
    def _dknob(s):
        try:m=os.stat(DKNOB).st_mtime_ns
        except OSError:return s.denv
        if m!=s.dkm:
            try:s.dms=float(open(DKNOB).read().split()[0]);s.dkm=m
            except Exception:s.dms=s.denv
        return s.dms
    def dec_wait(s,ms_over=None):
        ms=s._dknob()
        if ms_over is not None and ms!=0:ms=ms_over
        if ms==0:return
        if _async_on():
            if getattr(s,'DA',None) is None:s.DA=DecAsync(s)
            s.DA.submit(ms);return
        s.dseq+=1;n=s.dseq;torch.cuda.synchronize()
        ok,dt=s._dec_cond(n,ms)
        d=s.dst;d['n']+=1;d['tot']+=dt;d['to']+=0 if ok else 1;d['max']=max(d['max'],dt)
        if d['n']>=256:
            log.info('NestQuant dec-block rank %d: budget %s ms/layer, %d steps, timed out %d, mean %.2f ms, max %.1f ms',
                     s.rt.rank,ms,d['n'],d['to'],d['tot']/d['n'],d['max'])
            s.dst=dict(n=0,to=0,tot=0.,max=0.)
    def _dec_cond(s,n,ms):
        """after step n-1 finished on the GPU: wait (budget ms x layers) for its hits to be seen + the upgrade reads to land"""
        rt=s.rt;t0=time.perf_counter()
        dl=t0+(ms*len(s.L_)/1e3 if ms>0 else CAP_S);ok=False
        if s.dm is None and s.dpath and os.path.exists(s.dpath):
            try:s.dm=np.memmap(s.dpath,dtype=np.int64,mode='r',shape=(3,))
            except Exception:s.dm=None
        def wt(pred):
            while True:
                if not rt.in_iter and pred():return True
                r=dl-time.perf_counter()
                if r<=0:return False
                rt.wake.set();rt.cv.wait(min(r,0.001))
        with rt.cv:
            if rt.F is None:
                it0=rt.hl_n;ok=wt(lambda:rt.hl_n>=it0+2)
                if s.dm is not None and rt.log is not None:s.dm[1]=rt.log.gen;s.dm[2]=rt.log.off;s.dm[0]=n
            else:
                dm=s.dm;lg=rt.log
                ok=wt(lambda:dm is not None and dm[0]>=n and (lg.gen,lg.off)>=(int(dm[1]),int(dm[2])))
            if ok:ok=wt(lambda:not any(v[2]==4 for v in list(rt.X.ops.values())))
        return ok,(time.perf_counter()-t0)*1e3

DKNOB=os.environ.get('NQ_DEC_BLOCK_KNOB','/dev/shm/nq_dec_block')

# ---- NQ_DEC_ASYNC (eval knob, file /dev/shm/nq_dec_async "1" overrides env; default off): the decode-step wait without the
# host sync. execute_model records an event (= end of the work enqueued so far, i.e. the previous step), enqueues the
# csrc/nq_decwait.cu spin kernel (GPU waits for pinned flag >= n, or the step budget of GPU time) and goes on with its
# async prep; a waiter thread per rank does what dec_wait does after its synchronize (event.synchronize instead) and then
# sets flag = n. Same conditions, same budget (ms x MoE layers, <0 -> CAP_S), only the host no longer blocks.
AKNOB=os.environ.get('NQ_DEC_ASYNC_KNOB','/dev/shm/nq_dec_async');_AK=dict(m=None,v=os.environ.get('NQ_DEC_ASYNC','0')=='1')
def _async_on():
    try:m=os.stat(AKNOB).st_mtime_ns
    except OSError:m=-1
    if m!=_AK['m']:
        _AK['m']=m
        try:_AK['v']=(open(AKNOB).read().split()[0]=='1') if m!=-1 else os.environ.get('NQ_DEC_ASYNC','0')=='1'
        except Exception:_AK['v']=False
    return _AK['v']
class DecAsync:
    def __init__(s,P):
        import threading,queue
        from torch.utils.cpp_extension import load
        load(name='nq_decwait',sources=[os.path.join(os.path.dirname(os.path.abspath(__file__)),'csrc','nq_decwait.cu')],
             extra_cuda_cflags=['-O3'],is_python_module=False,verbose=False)
        s.P=P;s.q=queue.Queue();s.flag=torch.zeros(1,dtype=torch.int64,pin_memory=True);s.fv=s.flag.numpy()
        s.stat=torch.zeros(3,dtype=torch.int64,pin_memory=True);s.sv=s.stat.numpy();s.s0=s.sv.copy();s.hst=dict(n=0,to=0,tot=0.)
        s.th=threading.Thread(target=s._run,daemon=True,name='nq-decwait');s.th.start()
    def submit(s,ms):
        P=s.P;P.dseq+=1;n=P.dseq;ev=torch.cuda.Event();ev.record()
        tns=int(ms*len(P.L_)*1e6) if ms>0 else int(CAP_S*1e9)
        s.q.put((n,ev,ms));torch.ops.nq_decwait.spin(s.flag.data_ptr(),n,tns,s.stat.data_ptr())
        d=s.sv-s.s0
        if d[0]>=256:
            log.info('NestQuant dec-block async rank %d: budget %s ms/layer, %d steps, gpu timed out %d, gpu mean wait %.2f ms | host waiter mean %.2f ms, timed out %d',
                     P.rt.rank,ms,d[0],d[1],d[2]/d[0]/1e6,s.hst['tot']/max(1,s.hst['n']),s.hst['to'])
            s.s0=s.sv.copy();s.hst=dict(n=0,to=0,tot=0.)
    def _run(s):
        while True:
            n,ev,ms=s.q.get()
            try:
                ev.synchronize();ok,dt=s.P._dec_cond(n,ms)
                h=s.hst;h['n']+=1;h['tot']+=dt;h['to']+=0 if ok else 1
            except Exception:
                log.exception('NestQuant dec-block async waiter failed')
            finally:
                s.fv[0]=max(int(s.fv[0]),n)

# NQ_DEC_BLOCK_FIRSTN (eval knob, file /dev/shm/nq_dec_block_firstn "N [after_ms]" overrides env; 0/absent = off): the
# nq_dec_block budget applies only while a scheduled request is within its first N generated tokens
# (num_computed_tokens - prompt_len < N); after that the per-layer budget is after_ms (NQ_DEC_BLOCK_AFTER_MS, 0 = no wait). Same
# scheduler_output on every rank -> the same decision everywhere (keeps the rank-0/follower step counters in lockstep).
FNKNOB=os.environ.get('NQ_DEC_BLOCK_FIRSTN_KNOB','/dev/shm/nq_dec_block_firstn');_FN=dict(m=None,v=int(os.environ.get('NQ_DEC_BLOCK_FIRSTN','0')),after=0.0,pl={})
def _firstn_skip(so):
    try:m=os.stat(FNKNOB).st_mtime_ns
    except OSError:m=-1
    if m!=_FN['m']:
        _FN['m']=m
        try:
            f=open(FNKNOB).read().split() if m!=-1 else [os.environ.get('NQ_DEC_BLOCK_FIRSTN','0'),os.environ.get('NQ_DEC_BLOCK_AFTER_MS','0')]
            _FN['v']=int(float(f[0]));_FN['after']=float(f[1]) if len(f)>1 else 0.0
        except Exception:_FN['v']=0;_FN['after']=0.0
    pl=_FN['pl']
    for r in getattr(so,'scheduled_new_reqs',None) or []:
        pl[r.req_id]=len(r.prompt_token_ids or ())
    for r in getattr(so,'finished_req_ids',None) or ():pl.pop(r,None)
    N=_FN['v']
    if N<=0:return False
    c=so.scheduled_cached_reqs;g=[nc-pl.get(rid,0) for rid,nc in zip(c.req_ids,c.num_computed_tokens)]
    return bool(g) and min(g)>=N

def install_dec():
    """wraps GPUModelRunner.execute_model (every rank): decode-only steps (1..NQ_DEC_BLOCK_MAXTOK tokens) call dec_wait"""
    import vllm.v1.worker.gpu.model_runner as MR
    R=MR.GPUModelRunner
    if getattr(R,'_nq_dec_block',False):return
    orig=R.execute_model;MX=int(os.environ.get('NQ_DEC_BLOCK_MAXTOK','8'))
    def execute_model(self,scheduler_output,*a,**k):
        if not k.get('dummy_run',False) and getattr(scheduler_output,'scheduled_new_reqs',None):PI.new=True   # NQ_PRED_INPUTS new_request
        if not k.get('dummy_run',False) and 0<scheduler_output.total_num_scheduled_tokens<=MX:
            late=_firstn_skip(scheduler_output)
            if late and _FN['after']==0:return orig(self,scheduler_output,*a,**k)
            import nq_vllm
            P=getattr(nq_vllm.RT,'PFB',None)
            if P is not None:P.dec_wait(_FN['after'] if late else None)
        return orig(self,scheduler_output,*a,**k)
    R.execute_model=execute_model;R._nq_dec_block=True
    if os.environ.get('NQ_DEC_ASYNC_BUILD','1')=='1':     # JIT-build the spin kernel at boot (not on the first async step)
        try:
            from torch.utils.cpp_extension import load
            load(name='nq_decwait',sources=[os.path.join(os.path.dirname(os.path.abspath(__file__)),'csrc','nq_decwait.cu')],
                 extra_cuda_cflags=['-O3'],is_python_module=False,verbose=False)
        except Exception:log.exception('NestQuant dec-block: nq_decwait build failed (async wait unavailable)')
    log.info('NestQuant dec-block: execute_model wrapped (knob %s)',DKNOB)

# ---- NQ_PRED_INPUTS (eval knob, nq_vllm reads it): committed token ids + new_request for the predictor (think/answer
# segment state). GPUModelRunner.sample is wrapped on global rank 0: each step's sampled (= committed, rejection-sampled)
# token ids and their per-request counts are copied async into a pinned ring with an event; the leader thread drains the
# entries whose copy is done (pi_take). A step with new requests (scheduler_output.scheduled_new_reqs) sets PI.new.
import collections
class PI:
    NR=64;q=collections.deque(maxlen=48);ring=None;ri=0;new=False;err=0
def _grank():
    try:
        import torch.distributed as d
        return d.get_rank() if d.is_initialized() else 0
    except Exception:return 0
def pi_take():
    toks=[]
    while PI.q:
        tb,nb,ev,n,k=PI.q[0]
        if not ev.query():break
        PI.q.popleft()
        for i in range(n):toks.extend(tb[i,:max(0,min(int(nb[i]),k))].tolist())
    nr=PI.new;PI.new=False
    return toks,nr
def install_inputs():
    import vllm.v1.worker.gpu.model_runner as MR
    R=MR.GPUModelRunner
    if getattr(R,'_nq_pi',False):return
    orig=R.sample
    def sample(self,hidden_states,input_batch,grammar_output):
        F=_kldF();pre=F.cur if F is not None else None
        r=orig(self,hidden_states,input_batch,grammar_output)
        try:
            post=F.cur if F is not None else None
            if RH.acc is not None and _grank()==0:
                if pre is not None and pre!=post:_rh_flush(pre)
                if post is not None:_rh_rec(input_batch)
        except Exception:
            PI.err+=1
            if PI.err==1:log.exception('NestQuant rowhot set dump failed (further errors silent)')
        try:
            if _grank()==0:
                so,ns=r[0],r[1];t=so.sampled_token_ids
                if PI.ring is None:
                    PI.ring=[(torch.empty(64,8,dtype=torch.int64,pin_memory=True),torch.empty(64,dtype=torch.int64,pin_memory=True),torch.cuda.Event()) for _ in range(PI.NR)]
                n=t.shape[0];k=t.shape[1] if t.dim()>1 else 1
                if n<=64 and k<=8:
                    tb,nb,ev=PI.ring[PI.ri];PI.ri=(PI.ri+1)%PI.NR
                    tb[:n,:k].copy_(t.reshape(n,k),non_blocking=True)
                    if ns is not None:nb[:n].copy_(ns[:n],non_blocking=True)
                    else:nb[:n].fill_(1)
                    ev.record();PI.q.append((tb,nb,ev,n,k))
                if RH.acc is not None:_rh_step(ns,len(input_batch.logits_indices))
        except Exception:
            PI.err+=1
            if PI.err==1:log.exception('NestQuant pred-inputs: sample hook failed (further errors silent)')
        return r
    R.sample=sample;R._nq_pi=True
    log.info('NestQuant pred-inputs: sample wrapped (committed token ids -> predictor when NQ_PRED_INPUTS)')

# ---- NQ_ROWHOT (eval, rank 0): level-4 hot share over committed rows only vs all processed rows. nq_vllm.forward adds,
# per MoE layer, the level-4 routed count of each decode row into RH.acc[layer, row] (inside the graph); after the step's
# rejection sampling the first num_sampled rows are the committed positions (row 0 = last step's token, rows 1..j =
# accepted drafts). Accumulated on the GPU; a pinned snapshot every 32 steps feeds the io stats.
class RH:
    acc=None;li=None;ca=cc=na=nc=None;snap=None;k=0;sacc=scc=sca=None;cur=None;dsets=dpos=dnr=None;dk=0;L_=None
DR=int(os.environ.get('NQ_ROWHOT_RING','4096'))
def _kldF():
    try:
        import nq_kld;return nq_kld._F
    except Exception:return None
def _rh_rec(ib):
    """forced decode active: keep this step's level-4 set (every layer) + its row positions (live set replay dump)"""
    if RH.dsets is None:
        n=len(RH.li);RH.dsets=torch.zeros(DR,n,256,dtype=torch.bool,pin_memory=True)
        RH.dpos=torch.full((DR,8),-1,dtype=torch.int64,pin_memory=True);RH.dnr=torch.zeros(DR,dtype=torch.int64,pin_memory=True)
        RH.dids=torch.zeros(DR,n,8,8,dtype=torch.int16,pin_memory=True);RH.dsal=torch.zeros(DR,n,8,8,dtype=torch.float32,pin_memory=True)
    if RH.dk>=DR:return
    li=ib.logits_indices;n=min(int(li.shape[0]),8);k=RH.dk
    RH.dsets[k].copy_(RH.cur,non_blocking=True);RH.dids[k].copy_(RH.cids,non_blocking=True);RH.dsal[k].copy_(RH.csal,non_blocking=True);RH.dpos[k,:n].copy_(ib.positions[li[:n]],non_blocking=True);RH.dpos[k,n:]=-1
    RH.dnr[k]=n;RH._ns.append(k);RH.dk+=1
def _rh_flush(tag):
    torch.cuda.synchronize();k=RH.dk;out=os.environ.get('NQ_KLD_DIR','/dbg/kld')
    try:
        os.makedirs(f'{out}/{tag}',exist_ok=True)
        torch.save(dict(sets=RH.dsets[:k].clone(),pos=RH.dpos[:k].clone(),nrows=RH.dnr[:k].clone(),ns=RH.dns[:k].clone(),ids=RH.dids[:k].clone(),sal=RH.dsal[:k].clone(),
                        layers=list(RH.L_)),f'{out}/{tag}/sets.pt')
        log.info('NestQuant rowhot: %s/sets.pt %d steps',tag,k)
    finally:RH.dk=0;RH._ns=[]
def rh_init(L_,dev):
    RH.li={L:i for i,L in enumerate(L_)};n=len(L_)
    RH.acc=torch.zeros(n,8,dtype=torch.int32,device=dev)
    RH.ca=torch.zeros(n,dtype=torch.int64,device=dev);RH.cc=torch.zeros(n,dtype=torch.int64,device=dev)
    RH.na=torch.zeros((),dtype=torch.int64,device=dev);RH.nc=torch.zeros((),dtype=torch.int64,device=dev)
    RH.snap=torch.zeros(2*n+2,dtype=torch.int64,pin_memory=True);RH.L_=list(L_);RH._ns=[]
    RH.cur=torch.zeros(n,256,dtype=torch.bool,device=dev);RH.cids=torch.zeros(n,8,8,dtype=torch.int16,device=dev);RH.csal=torch.zeros(n,8,8,dtype=torch.float32,device=dev);RH.sacc=torch.zeros(n,8,2,dtype=torch.float32,device=dev)
    RH.sca=torch.zeros(n,2,dtype=torch.float64,device=dev);RH.scc=torch.zeros(n,2,dtype=torch.float64,device=dev)
    RH.ssnap=torch.zeros(4*n,dtype=torch.float64,pin_memory=True);RH.dns=torch.zeros(DR,dtype=torch.int64,pin_memory=True)
def _rh_step(ns,nrows):
    if ns is None or nrows<1 or nrows>8:
        RH.acc.zero_();RH.sacc.zero_()
        for k in RH._ns:RH.dns[k]=-1
        RH._ns=[];return
    r=torch.arange(8,device=RH.acc.device);va=(r<nrows);vc=(r<ns[0].to(torch.int64))&va
    RH.ca+=(RH.acc*va).sum(1);RH.cc+=(RH.acc*vc).sum(1);RH.na+=va.sum();RH.nc+=vc.sum();RH.acc.zero_()
    RH.sca+=(RH.sacc*va[None,:,None]).sum(1).double();RH.scc+=(RH.sacc*vc[None,:,None]).sum(1).double();RH.sacc.zero_()
    if RH._ns:
        for k in RH._ns:RH.dns[k:k+1].copy_(ns[:1],non_blocking=True)
        RH._ns=[]
    RH.k+=1
    if RH.k%32==0:
        RH.snap.copy_(torch.cat([RH.ca,RH.cc,RH.na[None],RH.nc[None]]),non_blocking=True)
        RH.ssnap.copy_(torch.cat([RH.sca.flatten(),RH.scc.flatten()]),non_blocking=True)
def rh_stats():
    if RH.snap is None:return {}
    v=RH.snap.tolist();n=len(RH.li)
    w=RH.ssnap.tolist()
    return dict(rh_all=v[:n],rh_com=v[n:2*n],rh_rows_all=v[2*n],rh_rows_com=v[2*n+1],
                rs_all_hot=w[0:2*n:2],rs_all_tot=w[1:2*n:2],rs_com_hot=w[2*n::2],rs_com_tot=w[2*n+1::2])
