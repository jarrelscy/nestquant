"""NestQuant experts in vLLM (GLM-5.3 ARVQ serving image). Imported by the patched nvfp4_arvq_hybrid.py when NQ_REPACK
is set: every MoE layer that has resident planes in NQ_REPACK/res/rank{r}/L{L}.pt (and records in rank{r}.bin) runs
the NestQuant kernel instead of ARVQ; other layers keep ARVQ, so a partly fitted model serves.

Per worker process (one TP rank):
  - NestQuantMoEMethod registers the checkpoint's ARVQ expert parameter names as empty tensors with a no-op loader
    (the ARVQ expert tensors are read and dropped), then loads the rank's resident planes (level 2 rows).
  - after the last NQ layer is loaded, Runtime.start(): fixed set (26/layer, update 02) read from the record file
    into a resident pool at level 4; slot pool of NQ_SLOTS_PER_LAYER x layers records; scheduler (EMA 512, refresh 64,
    51 floating/layer, 6 GB/s cap, start = floating_default) + executor (io_uring engine) + a host thread that turns
    the kernel's routing hits into level ops. Table rows change only through the Mailbox, whose apply() runs at the
    start of every layer call (inside CUDA graphs too).
  - forward = torch custom op nq::moe (opaque to torch.compile): <= 8 tokens one kernel call; >= NQ_PF_MIN (384) tokens the
    prefill path (moe.MoELayer.prefill: routed experts decoded once + grouped GEMMs); in between chunks of 8.
Env: NQ_HOME (repo, default /nq), NQ_REPACK (record + resident dir), NQ_SLOTS_PER_LAYER (56), NQ_CAP_GBPS (0 = uncapped: drive and slot pool limit; else aggregate GB/s over the TP ranks),
NQ_TOK_PER_S (111), NQ_STREAM (1; 0 = fixed set only, no streaming), NQ_POLL_MS (4), NQ_LAYERS (e.g. 3-18, default all
present), NQ_PREDICTOR (floating-set predictor: ema | gbdt, default scheduler.DEFAULT_PREDICTOR; only the scheduling rank
runs it; NQ_GBDT_MODE / NQ_GBDT_SCALE / NQ_GBDT_BAND see streaming/scheduler.py; NQ_GBDT_SCALE=mps makes rank 0 export per-expert decode
salience w^2*|x|^2 via nqsal.cu, rsf = hf routed_scaling_factor or NQ_SAL_RSF), NQ_SESSION_RESTORE / NQ_SR_* (per-session floating-set restore for interleaved sessions, rank 0 decides, see nq_session.py; hooks Worker.execute_model for request arrival), NQ_LGB_PATH (dir with lightgbm + narwhals + scipy, appended to sys.path; serve_nq.sh mounts it at /nqlgb). Ranks schedule independently from their own hit counters (the counts agree, the timing of a refresh can
differ by a step between ranks, so for a short while an expert can be at level 4 on some ranks and level 2 on others;
each rank's shard is a valid level-2 or level-4 weight either way)."""
import os,sys,json,time,threading,logging,functools
import numpy as np,torch
NQ_HOME=os.environ.get('NQ_HOME','/nq')
for _p in (NQ_HOME+'/streaming',NQ_HOME+'/sm120'):
    if _p not in sys.path:sys.path.insert(0,_p)
if os.environ.get('NQ_LGB_PATH') and os.environ['NQ_LGB_PATH'] not in sys.path:sys.path.append(os.environ['NQ_LGB_PATH'])
try:
    from vllm.logger import init_logger;log=init_logger('vllm.nestquant')
except Exception:log=logging.getLogger('nestquant')
# kernel tiling per token count (sm120/bench_real.py on real layers: best cfg per B, 2-11% over one fixed cfg)
CFG_GU=[None,[1,8,4],[1,8,6],[1,8,6],[1,8,6],[1,8,6],[1,8,6],[1,8,12],[1,8,12]]
CFG_DN=[None,[1,8,4],[1,8,2],[1,8,4],[1,8,4],[1,8,4],[1,8,4],[1,8,4],[1,8,4]]
NE=256;TOPK=8;BMAX=8;NF=51;CHECK=os.environ.get('NQ_CHECK','0')=='1'
# prefill (T >= NQ_PF_MIN tokens, not capturing): moe.MoELayer.prefill; below it (and inside graph capture) the decode
# kernel in 8-token slices. NQ_PF=0 disables; while the file NQ_PF_OFF (default /dev/shm/nq_pf_off) exists the slice
# loop runs (in-boot A/B). Scratch (moe.pf_scratch, NQ_PF_ROWS x NQ_PF_G) is allocated on the first prefill call, i.e.
# in vLLM's profile run, so the KV budget accounts for it.
PF=os.environ.get('NQ_PF','1')!='0';PF_MIN=int(os.environ.get('NQ_PF_MIN','384'));PF_OFF=os.environ.get('NQ_PF_OFF','/dev/shm/nq_pf_off')
# prefill expert-level adaptation (NQ_PREFILL_ADAPT=lookahead|chunk, NQ_LA_MEASURE=1): see nq_lookahead.py
import nq_lookahead as LAH,nq_session as SRM
RSF=None          # routed_scaling_factor the MoE runner applies after the experts (topk_weights here exclude it); set in create_weights
if LAH.MODE or LAH.MEAS:LAH.install()
if os.environ.get('NQ_KLD_HOOK','0')=='1':   # nq-kld full-vocab prompt-logprob dump (eval only, nq_kld.py)
    import nq_kld;nq_kld.install();nq_kld.install_force()
ARVQ_NAMES=('hyb_kind',)+tuple(f'arvq_{p}_{k}' for p in ('w13','w2') for k in ('packed','scales','codebooks','global'))+\
    tuple(f'nvfp4_{p}_{k}' for p in ('w13','w2') for k in ('packed','bscale','scale2'))

def _layers_env():
    s=os.environ.get('NQ_LAYERS')
    if not s:return None
    a,b=(s.split('-')+[s])[:2];return set(range(int(a),int(b)+1))

def available(L,rank):
    """NQ serves layer L on this rank iff its resident planes and records are both written."""
    rp=os.environ.get('NQ_REPACK')
    if not rp:return False
    only=_layers_env()
    if only is not None and L not in only:return False
    ip=f'{rp}/rank{rank}.json'
    if not (os.path.exists(f'{rp}/res/rank{rank}/L{L}.pt') and os.path.exists(ip)):return False
    return str(L) in json.load(open(ip))['layers']

def _stack_limit(need=2048):
    """The persistent kernels use ~1.1 KB of stack per thread (> the 1 KB default). The driver grows the limit at
    launch only if it can allocate the local-memory pool then; under vLLM the GPU is full by the first MoE call, so
    raise it at load time."""
    import ctypes
    lib=None
    for n in ('libcudart.so.13','libcudart.so.12','libcudart.so'):
        try:lib=ctypes.CDLL(n);break
        except OSError:pass
    if lib is None:
        import glob;c=glob.glob(os.path.dirname(torch.__file__)+'/../nvidia/cu*/lib/libcudart.so*')+glob.glob(os.path.dirname(torch.__file__)+'/../nvidia/cuda_runtime/lib/libcudart.so*')
        lib=ctypes.CDLL(c[0])
    v=ctypes.c_size_t();lib.cudaDeviceGetLimit(ctypes.byref(v),0);old=v.value
    if old<need:
        r=lib.cudaDeviceSetLimit(0,ctypes.c_size_t(need));assert r==0,f'cudaDeviceSetLimit(stack,{need}) failed: {r}'
    lib.cudaDeviceGetLimit(ctypes.byref(v),0);return old,v.value

class Runtime:
    """Per-process registry of the NQ layers + the streaming machinery."""
    def __init__(s):
        s.expect=set();s.lay={};s.started=False;s.thread=None;s.stop=False;s.lock=threading.Lock();s.err=None
        s.cv=threading.Condition();s.ncap=0;s.in_iter=False;s.wake=threading.Event();s.LA=None;s.SAL=None;s.SR=None;s.CAP=None
        s.PB=None;s.pb_pause=0      # nq-prefill prefill-borrow (nq_pb.py), NQ_PREFILL_BORROW=1 only
    def expect_layer(s,L):s.expect.add(L)
    def add_layer(s,L,rank,tp,dev):
        import resident as RS
        from moe import MoELayer,Mailbox,entry
        if not s.lay:log.info('NestQuant rank %d: CUDA stack limit %d -> %d B/thread',rank,*_stack_limit())
        t=time.time();ex,H,I=RS.load(f"{os.environ['NQ_REPACK']}/res/rank{rank}/L{L}.pt",dev)
        M=MoELayer(NE,H,I,Bmax=BMAX,dev=dev);MB=Mailbox(M)
        for E,x in ex.items():M.table[E].copy_(entry(x,2))
        hits=torch.zeros(NE,dtype=torch.int32).pin_memory()
        if os.environ.get('NQ_HITS','1')!='0':M.hits_ptr=hits.data_ptr()
        s.lay[L]=dict(M=M,MB=MB,ex=ex,hits=hits,H=H,I=I);s.rank,s.tp,s.dev=rank,tp,dev
        log.info('NestQuant L%d rank %d: resident planes loaded in %.1fs',L,rank,time.time()-t)
        if set(s.lay)>=s.expect and not s.started:s.start()
    def start(s):
        import stream_engine as SE,scheduler as SC,executor as EX,fixed_set as FS,p4rec as PR
        from moe import entry
        s.started=True;rp=os.environ['NQ_REPACK'];L_=sorted(s.lay);dev=s.dev
        s.rf=rf=SE.RankFile(rp,s.rank);rb=rf.rb
        fx,src,_=FS.load(layers=L_)
        _JOINT=os.environ.get('NQ_PREDICTOR','') in ('joint','jf','tf')   # tf = nq-tfpred transformer (same k0 layout); jF joint predictor: k0 layout (no fixed set, all floating)
        nf=77 if _JOINT else NF
        if _JOINT:fx={L:[] for L in L_};src='joint-k0'
        if os.environ.get('NQ_STREAM','1')=='0' and os.environ.get('NQ_STATIC_FLOAT','0')=='1' and not _JOINT:   # nq-kld static arm: fixed set U floating_default (nqfloat0), no streaming
            fj=json.load(open(NQ_HOME+'/threads/22-boundary-experts/fixed_set.json'))
            fx={L:list(fx[L])+[int(x) for x in np.argsort(-np.where(np.isin(np.arange(NE),fx[L]),-1,np.array(fj['n_routed'][str(L)])))[:NF]] for L in L_};src+='+floating_default%d'%NF
        # fixed set: records -> resident pool, level 4 rows
        nfix=sum(len(fx[L]) for L in L_);s.fixpool=torch.empty(nfix,rb,dtype=torch.uint8,device=dev)
        buf=torch.empty(rb,dtype=torch.uint8).pin_memory();fd=os.open(rf.path,os.O_RDONLY);i=0;t=time.time();bad=set()
        try:
            for L in L_:
                d=s.lay[L]
                for E in fx[L]:
                    try:n=os.preadv(fd,[memoryview(buf.numpy())],rf.rec(L,E)*rb)
                    except OSError:n=-1
                    if n!=rb:bad.add((L,E));i+=1;continue      # SSD unreadable / short: the expert stays at level 2
                    s.fixpool[i].copy_(buf);d['M'].table[E].copy_(PR.row(d['ex'][E],rf.lay,s.fixpool[i].data_ptr(),entry));i+=1
        finally:os.close(fd)
        torch.cuda.synchronize(dev)
        if bad:
            log.warning('NestQuant rank %d: %d fixed-set records unreadable, those experts stay at level 2',s.rank,len(bad))
            fx={L:[E for E in fx[L] if (L,E) not in bad] for L in fx};nfix-=len(bad)
        for L in L_:s.lay[L]['table0']=s.lay[L]['M'].table.clone()
        log.info('NestQuant rank %d: %d layers, fixed set (%s) %d experts at level 4 (%.1f GiB) in %.1fs',
                 s.rank,len(L_),src,nfix,nfix*rb/2**30,time.time()-t)
        if os.environ.get('NQ_STREAM','1')=='0':
            s.L_=L_;fm=np.zeros((len(L_),NE),bool)
            for i,L in enumerate(L_):fm[i,list(fx[L])]=True
            s.thread=threading.Thread(target=s.share_loop,args=(fm,),name='nq-share',daemon=True);s.thread.start();return
        nslot=int(os.environ.get('NQ_SLOTS_PER_LAYER','56'))*len(L_)
        T22=NQ_HOME+'/threads/22-boundary-experts/fixed_set.json';fj=json.load(open(T22))
        dflt={L:[int(x) for x in np.argsort(-np.where(np.isin(np.arange(NE),fx[L]),-1,np.array(fj['n_routed'][str(L)])))[:nf]] for L in L_}
        lead=os.environ.get('NQ_LEADER','1')!='0' and s.tp>1
        _steps=(s.rank==0 or not lead)   # this rank runs the real predictor; the others use 'ema'
        if _JOINT and _steps and os.environ.get('NQ_PREDICTOR')=='tf':   # nq-tfpred multi-window transformer (threads/36-tfpred)
            import nq_tfpred_gpu as TFP
            pred=TFP.TFGPUPredictor(L_,fx,os.environ.get('NQ_TF_CKPT',NQ_HOME+'/threads/36-tfpred/tf_ids.pt'),n_float=nf,
                                    hm=float(os.environ.get('NQ_JOINT_HM','0.7')),device=dev,graph=os.environ.get('NQ_TF_GRAPH','1')=='1')
        elif _JOINT and _steps:
            _jd='/nqpred/joint'
            if _jd not in sys.path:sys.path.insert(0,_jd)
            import gpu_predictor as GJ
            pred=GJ.GPUJointPredictor(L_,fx,os.environ.get('NQ_JOINT_NET','/nqpred/joint/jF.pt'),n_float=nf,
                                      hm=float(os.environ.get('NQ_JOINT_HM','0.7')),device=dev,
                                      v2_model=os.environ.get('NQ_JOINT_V2','/nqpred/joint/v2_sal_tweedie1.5.txt'),
                                      graph=os.environ.get('NQ_JOINT_GRAPH','0')=='1')
        else:
            pred=(os.environ.get('NQ_PREDICTOR') or SC.DEFAULT_PREDICTOR) if _steps else 'ema'   # followers never step S
        _SCH=SC.Scheduler if not os.environ.get('NQ_SCHED') else __import__('scheduler_tap').make_scheduler   # NQ_SCHED=tap: streaming/scheduler_tap.py (nq-tfpred 93beb7f)
        s.S=_SCH(L_,fx,dflt,rb*s.tp,NE=NE,n_float=nf,slots=nslot,cap_GBps=float(os.environ.get('NQ_CAP_GBPS','0')) or 1e6,
                         tok_per_s=float(os.environ.get('NQ_TOK_PER_S','111')),predictor=pred)
        log.info('NestQuant rank %d: floating-set predictor %s',s.rank,s.S.predictor_name)
        s.S.io_all=lambda:io_all(s)    # nq-io: live per-rank I/O stats for scheduler policies ({} unless NQ_IOSTATS>0 and TP>1)
        IO=_io_cfg(s.rank,rp,rb)
        s.X=EX.RankExecutor(rf,{L:(s.lay[L]['M'],s.lay[L]['MB'],s.lay[L]['ex']) for L in L_},nslot,n_host=IO['n_host'],qd=IO['qd'],device=dev.index,
                               shadow=os.environ.get('NQ_SHADOW','0')=='1',**IO['kw'])
        if IO['log']:log.info('NestQuant rank %d: nq-io %s, RAM tier %d records (%.1f GiB)',s.rank,IO['log'],s.X.tier_n,s.X.tier_n*rb/2**30)
        if s.X.tier_n:_tier_watch(s)
        init=[(L,E) for L in L_ for E in dflt[L] if E not in fx[L]][:nslot]
        for L,E in init:s.S.state[s.S.li[L],E]=1
        s.X.apply(init,[],s.S)
        s.F=s.log=None
        if os.environ.get('NQ_LEADER','1')!='0' and s.tp>1:     # rank 0 schedules, the other ranks replay its ops
            import oplog as OL
            # /dev/shm is the host's and container pids repeat across boots: key the log by the parent's start time too,
            # so a follower can never open (and replay) a previous boot's log before the leader has created this one
            pp=os.getppid();st=open(f'/proc/{pp}/stat').read().rsplit(')',1)[1].split()[19]
            s.log=OL.OpLog(f'/dev/shm/nq_oplog_{pp}_{st}.bin',writer=s.rank==0)
            if s.rank:
                coal=os.environ.get('NQ_FOLLOW_COALESCE','0')=='1'
                if HOSTLOOP=='cpp':s.F=OL.CppFollower(s.X,s.log,L_,coalesce=coal)        # nq-io upgrade 5
                else:s.F=(OL.CoalescingFollower if coal else OL.Follower)(s.X,s.log)
                s.F.mark_busy(init)
                if RANK_SHARE and s.F.mask is None:s.F.mask=np.zeros((len(L_),NE),bool);s.F.li={L:i for i,L in enumerate(L_)}
                if CHECK or os.environ.get('NQ_FOLLOW_CHECK','0')=='1':s.F.enable_check(init)
            s.iokey=f'/dev/shm/nq_io_{pp}_{st}'
        log.info('NestQuant rank %d: %d slots (%.1f GiB), floating_default %d upgrades issued',s.rank,nslot,nslot*rb/2**30,len(init))
        if s.F is None and s.S.wants_sal:        # GBDT x mps128: rank 0 exports per-expert decode salience next to the hits
            import build as BLD
            s.SAL=BLD.get_sal();rsf=float(os.environ.get('NQ_SAL_RSF') or RSF or 1.0)
            for L in L_:
                d=s.lay[L];d['sal_acc']=torch.zeros(NE,dtype=torch.float64,device=dev)
                d['sal_host']=torch.zeros(NE,dtype=torch.float64).pin_memory();d['sal_rsf']=rsf
            log.info('NestQuant rank %d: decode salience export on (%s, rsf %.3f)',s.rank,s.S.predictor_name,rsf)
        _gate_captures(s)
        if LAH.MODE or LAH.MEAS:s.LA=LAH.LA(s)
        if s.F is None and s.rank==0:s.SR=SRM.SessionRestore(s);_hook_sched(s)   # per-session floating-set restore (nq_session.py)
        if os.environ.get('NQ_PREFILL_BORROW','0')=='1':     # nq-prefill: KV blocks lent to the slot pool during long prefills
            if HOSTLOOP=='cpp':log.warning('NestQuant prefill-borrow: NQ_HOSTLOOP=cpp not supported, off')
            else:
                import nq_pb;s.PB=nq_pb.install(s)
        if os.environ.get('NQ_TFCAP') and s.F is None and s.rank==0:   # nq-tfpred decode-trace capture (off unless set)
            import nq_tfcap;s.CAP=nq_tfcap.install(s,L_,s.lay[L_[0]]['H'],dev)
        s.L_=L_;s.thread=threading.Thread(target=s.loop,name='nq-stream',daemon=True);s.thread.start()
    def _sr(s,f,*a,dflt=None):
        """session restore is an optimization: any error turns it off (pin dropped), streaming goes on"""
        try:return f(*a)
        except Exception:
            log.exception('NestQuant session restore failed, turned off');s.SR=None;s.S.pin=None;return dflt
    def share_loop(s,lv4):
        """NQ_STREAM=0: log the level-4 share of routed slots of the fixed set (same format as the streaming loop)."""
        H=[s.lay[L]['hits'].numpy() for L in s.L_];prev=np.stack(H).astype(np.int64);s4=st=0;n=0
        while not s.stop:
            time.sleep(1.);n+=1;cur=np.stack(H).astype(np.int64);c=cur-prev;prev=cur;s4+=int(c[lv4].sum());st+=int(c.sum())
            if st and n%60==0:
                log.info('NestQuant rank %d: level-4 hit share %.4f (%d/%d routed slots)',s.rank,s4/st,s4,st);s4=st=0
    def loop(s):
        try:
            torch.cuda.set_device(s.dev);ms=float(os.environ.get('NQ_POLL_MS','4'))/1e3
            issue=os.environ.get('NQ_ISSUE','1')!='0'    # 0: schedule but never issue ops (level set stays floating_default; A/B only)
            H=[s.lay[L]['hits'].numpy() for L in s.L_];prev=np.zeros((len(s.L_),NE),np.int64);last=time.time();tb=0.;nit=0;s4=st=0
            SH=[s.lay[L]['sal_host'].numpy() for L in s.L_] if 'sal_host' in s.lay[s.L_[0]] else None;sprev=np.zeros((len(s.L_),NE))
            fsh=RANK_SHARE and s.F is not None and s.F.mask is not None;fs4=fst=0;fprev=np.zeros((len(s.L_),NE),np.int64)   # nq-io follower share
            sp4=[0,0];spt=[0,0];s.shc=[0,0,0,0]   # nq-io NQ_RANK_SHARE: level-4 share split [decode, prefill] (poll interval > PF_NTOK tokens = prefill)
            iot=time.time();iok=getattr(s,'iokey',None) if IOSTATS else None;fct=0.;s.hl_t=0.;s.hl_n=0
            while not s.stop:
                s.wake.wait(ms);s.wake.clear()      # prefill adapt wakes the loop as soon as a layer's router stats are queued
                with s.cv:
                    if s.PB is not None:s.cv.wait_for(lambda:not s.pb_pause)   # prefill-borrow / reclaim in progress
                    s.in_iter=True;cap=s.ncap>0
                t0=time.perf_counter()
                try:
                    if s.F is not None:
                        s.X.poll(s.F,issue=not cap);s.F.step(issue=issue and not cap)
                        if s.F.chk is not None and time.time()-fct>1.:
                            fct=time.time();r=s.F.check()
                            if r is not None and not r[0] and s.F.stats['check_bad']<=5:
                                log.error('NestQuant rank %d: follower != leader log when quiescent: %d missing %d extra (%s)',s.rank,r[1],r[2],s.F.check_msg)
                        if fsh and not cap:          # this rank's served level-4 share (its own landed set, not the leader's)
                            cur=np.stack(H).astype(np.int64);c=cur-fprev;fprev=cur
                            if c.sum()>0:                  # every interval (a decode step's later layers often land after layer 0's poll)
                                a4=int(c[s.S.fixed|s.F.mask].sum());at=int(c.sum());fs4+=a4;fst+=at
                                k=_pfk(c);sp4[k]+=a4;spt[k]+=at;s.shc[k]+=a4;s.shc[2+k]+=at
                    else:
                        s.X.poll(s.S,issue=not cap)
                    if s.LA is not None and s.F is None and not cap and issue:s.LA.service(s.S,s.X,s.log)
                    if s.SR is not None and not cap and issue:s._sr(s.SR.service,s.S,s.X,s.log)
                    if s.F is None and not cap:        # hits counted during a capture are dropped with it (warmup inputs)
                        cur=np.stack(H).astype(np.int64);c=cur-prev;prev=cur;ntok=int(c[0].sum())//TOPK
                        if RANK_SHARE and ntok==0 and c.sum()>0:   # nq-io share: intervals without layer-0 hits (prod path ignores them)
                            lv=s.S.fixed|(s.S.state==2);k=_pfk(c);a4=int(c[lv].sum());at=int(c.sum());sp4[k]+=a4;spt[k]+=at;s.shc[k]+=a4;s.shc[2+k]+=at
                        if ntok>0:
                            sal=None
                            if SH is not None:scur=np.stack(SH);sal=np.maximum(scur-sprev,0.);sprev=scur   # cumulative fp64, diffed like the hits
                            lv=s.S.fixed|(s.S.state==2);s4+=int(c[lv].sum());st+=int(c.sum())   # share at the levels served this step
                            if RANK_SHARE:k=_pfk(c);a4=int(c[lv].sum());at=int(c.sum());sp4[k]+=a4;spt[k]+=at;s.shc[k]+=a4;s.shc[2+k]+=at
                            if s.SR is not None and issue:ntok=s._sr(s.SR.on_counts,s.S,c,ntok,lv,dflt=ntok)   # handover / prefill residue / window share
                            ups,downs=s.S.step(c,ntok,sal=sal)
                            if issue:
                                if s.PB is not None:ups=s.X.pool_tag(ups)
                                s.X.apply(ups,downs,s.S)
                                if s.log is not None:s.log.put(ups,downs)
                finally:
                    with s.cv:s.in_iter=False;s.cv.notify_all()
                    _dt=time.perf_counter()-t0;tb+=_dt;nit+=1;s.hl_t+=_dt;s.hl_n+=1   # hl_*: cumulative (io stats)
                if iok is not None and time.time()-iot>=IOSTATS:
                    iot=time.time();_io_publish(s,iok)
                if time.time()-last>60:
                    last=time.time();lv=s.S.level()
                    if s.F is not None:
                        log.info('NestQuant rank %d: follower, level-4 floating %d, stats %s, backlog %d',s.rank,s.F.level_count(),s.F.stats,s.F.backlog())
                    if st:log.info('NestQuant rank %d: level-4 hit share %.4f (%d/%d routed slots)',s.rank,s4/st,s4,st);s4=st=0
                    if fst:log.info('NestQuant rank %d: follower level-4 hit share %.4f (%d/%d routed slots)',s.rank,fs4/fst,fs4,fst);fs4=fst=0
                    if spt[0] or spt[1]:
                        log.info('NestQuant rank %d: level-4 hit share split decode %.4f (%d slots) prefill %.4f (%d slots)',s.rank,
                                 sp4[0]/max(spt[0],1),spt[0],sp4[1]/max(spt[1],1),spt[1])
                        sp4=[0,0];spt=[0,0]
                    log.info('NestQuant rank %d: host loop %.0f us/iter x %d iters (%.1f%% of wall)',s.rank,tb/max(nit,1)*1e6,nit,tb/60*100);tb=0.;nit=0
                    log.info('NestQuant rank %d: level-4 experts %d/%d, ups %d downs %d, read errors %d, op p50 %.1f ms',s.rank,
                             int((lv==4).sum()),lv.size,s.S.stats['ups'],s.S.stats['downs'],s.S.stats.get('read_errors',0),
                             float(np.median(s.X.lat[-256:]))*1e3 if s.X.lat else -1)
                    if s.F is None and s.S.P is not None:log.info('NestQuant rank %d: predictor %s stats %s',s.rank,s.S.predictor_name,getattr(s.S.P,'stats',{}))
        except Exception as e:           # streaming stops; every expert keeps its current (valid) row
            s.err=e;log.exception('NestQuant streaming thread stopped')

# ---- nq-io (I/O upgrades; every flag off = the original serve) ----
# NQ_IO_MODE      '' (one drive, NQ_REPACK) | dual (each read to the less-busy of NQ_REPACK / NQ_REPACK_ALT) |
#                 split (ranks in NQ_IO_SPLIT_RANKS, default 2,3, read only NQ_REPACK_ALT)
# NQ_REPACK_ALT   identical copy of the record files on the second drive (rank*.json + artifact_stamp.json must match)
# NQ_IO_QD        reads in flight per rank per drive (default 8 = original); NQ_IO_QD_ALT for the alt drive (default NQ_IO_QD)
# NQ_IO_NHOST     pinned bounce/LRU entries (default max(64, 4 x total QD))
# NQ_RAMTIER_GB   pinned host-RAM tier per rank (GB, 0 = off); NQ_RAMTIER_LIST json [[L, E], ...] hottest first;
#                 NQ_RAMTIER_MINAVAIL_GB (30; host memguard kills at 22): never load past, and drop the tier when MemAvailable falls below it
#                 (also dropped while /dev/shm/nq_tier_drop exists)
# NQ_FOLLOW_COALESCE=1  followers replay by net effect (oplog.CoalescingFollower)
# NQ_HOSTLOOP=cpp  leader Scheduler.step array work + follower replay in C++ (streaming/nqhost.cpp, bit-exact)
# NQ_FOLLOW_CHECK=1 (or NQ_CHECK=1) followers check, whenever quiescent, landed set == leader log's set (stats check_ok/bad)
# NQ_RANK_SHARE=1 followers log their own served level-4 share; NQ_IOSTATS=<s> every rank writes its io_stats() JSON
#                 to /dev/shm/nq_io_<boot>_r<rank>.json every <s> seconds (Runtime.io_all() reads them all)
RANK_SHARE=os.environ.get('NQ_RANK_SHARE','0')=='1';PF_NTOK=int(os.environ.get('NQ_SHARE_PF_NTOK','32'));HOSTLOOP=os.environ.get('NQ_HOSTLOOP','py');IOSTATS=float(os.environ.get('NQ_IOSTATS','0') or 0)
def _pfk(c):return int(int(c.sum(1).max())//TOPK>PF_NTOK)   # nq-io share: 1 = prefill interval (busiest layer saw > PF_NTOK tokens)
def _memavail_gb():
    for ln in open('/proc/meminfo'):
        if ln.startswith('MemAvailable:'):return int(ln.split()[1])/2**20
    return 0.
def _io_cfg(rank,rp,rb):
    mode=os.environ.get('NQ_IO_MODE','');qd=int(os.environ.get('NQ_IO_QD','8'));kw={};msg=[]
    alt=os.environ.get('NQ_REPACK_ALT','')
    if mode in ('dual','split'):
        ok=bool(alt) and all(open(f'{rp}/{f}','rb').read()==open(f'{alt}/{f}','rb').read() for f in (f'rank{rank}.json','artifact_stamp.json'))
        if not ok:log.warning('NestQuant rank %d: NQ_IO_MODE=%s but %s does not match %s, one drive',rank,mode,alt,rp);mode=''
        elif os.path.getsize(f'{alt}/rank{rank}.bin')!=os.path.getsize(f'{rp}/rank{rank}.bin'):
            log.warning('NestQuant rank %d: %s/rank%d.bin size differs, one drive',rank,alt,rank);mode=''
    qa=int(os.environ.get('NQ_IO_QD_ALT') or qd) if mode=='dual' else 0
    if mode=='dual':kw.update(alt_path=f'{alt}/rank{rank}.bin',qd_alt=qa);msg.append(f'dual drive qd {qd}+{qa}')
    elif mode=='split' and str(rank) in os.environ.get('NQ_IO_SPLIT_RANKS','2,3').split(','):
        RT.rf.path=f'{alt}/rank{rank}.bin';msg.append(f'split: reads {RT.rf.path}')
    elif mode=='split':msg.append('split: reads primary')
    if qd!=8 and mode!='dual':msg.append(f'qd {qd}')
    nh=int(os.environ.get('NQ_IO_NHOST') or max(64,4*(qd+qa)))
    gb=float(os.environ.get('NQ_RAMTIER_GB','0') or 0)
    if gb>0:
        lst=json.load(open(os.environ['NQ_RAMTIER_LIST']));lay=set(RT.lay)
        lst=[(int(L),int(E)) for L,E in lst if int(L) in lay]
        floor=float(os.environ.get('NQ_RAMTIER_MINAVAIL_GB','30'))
        av=_memavail_gb();tp=max(RT.tp,1)
        fit=max(0.,(av-floor-4.)/tp)                     # all ranks load at once: leave floor + 4 GB margin after all of them
        g=min(gb,fit);n=min(len(lst),int(g*1e9//rb))
        if g<gb:log.warning('NestQuant rank %d: RAM tier %.1f GB asked, MemAvailable %.1f GiB -> %.1f GB',rank,gb,av,g)
        if n>0:kw['tier']=lst[:n];msg.append(f'RAM tier {n} recs')
    return dict(qd=qd,n_host=nh,kw=kw,log=', '.join(msg))
def _tier_watch(rt):
    floor=float(os.environ.get('NQ_RAMTIER_MINAVAIL_GB','30'))
    def run():
        while not rt.stop:
            time.sleep(0.5);av=_memavail_gb()
            if av<floor or os.path.exists('/dev/shm/nq_tier_drop'):
                rt.X.eng.tier_drop();log.warning('NestQuant rank %d: RAM tier DROPPED (MemAvailable %.1f GiB, floor %.0f)',rt.rank,av,floor);return
    threading.Thread(target=run,name='nq-tierwatch',daemon=True).start()
def _io_publish(rt,key):
    try:
        d=rt.X.io_stats('publish');d.update(rank=rt.rank,t_wall=time.time(),backlog=rt.F.backlog() if rt.F is not None else 0,
                                   follower=dict(rt.F.stats) if rt.F is not None else None,memavail_gb=_memavail_gb())
        st=rt.F.stats if rt.F is not None else rt.S.stats   # cumulative: host loop seconds / iterations, level ops issued
        d.update(share4=getattr(rt,'shc',None),hostloop=HOSTLOOP,hl_s=getattr(rt,'hl_t',0.),hl_iters=getattr(rt,'hl_n',0),ops_issued=int(st['ups'])+int(st['downs']))
        tmp=f'{key}_r{rt.rank}.json.tmp';open(tmp,'w').write(json.dumps(d));os.replace(tmp,f'{key}_r{rt.rank}.json')
    except Exception:log.exception('NestQuant io stats publish failed')
def io_all(rt=None):
    """leader-side API: {rank: io_stats dict (+ backlog, follower stats, memavail)} from every rank's last publish"""
    rt=rt or RT;key=getattr(rt,'iokey',None);out={}
    if key is None:return out
    for r in range(max(rt.tp,1)):
        try:out[r]=json.load(open(f'{key}_r{r}.json'))
        except (OSError,ValueError):pass
    return out

def _hook_sched(rt):
    """Worker.execute_model(scheduler_output) -> rt.SR.on_sched first (request arrival = its first step, before the
    step's LMCache load and forward): the session key is known as early as the worker can see the request"""
    try:from vllm.v1.worker import gpu_worker as GW
    except Exception as e:log.warning('NestQuant session restore: no gpu_worker (%s), off',e);rt.SR=None;return
    W=GW.Worker
    if getattr(W,'_nq_sr',False):return
    ex0=W.execute_model
    def execute_model(self,scheduler_output,*a,**k):
        sr=RT.SR
        if sr is not None:
            try:sr.on_sched(scheduler_output)
            except Exception:log.exception('NestQuant session restore: hook failed, off');RT.SR=None
        return ex0(self,scheduler_output,*a,**k)
    functools.update_wrapper(execute_model,ex0);W.execute_model=execute_model;W._nq_sr=True

def _gate_captures(rt):
    """The engine thread calls cudaEventQuery / cudaMemcpyAsync while it has ops in flight; any of those during a
    global-mode CUDA graph capture (vLLM's) invalidates the capture. So every capture_begin waits until the host loop
    is between iterations and the engine is idle, and no new op is issued until capture_end."""
    G=torch.cuda.CUDAGraph
    if getattr(G,'_nq_gated',False):return
    b0,e0=G.capture_begin,G.capture_end
    def begin(g,*a,**k):
        with rt.cv:
            rt.ncap+=1
            if not rt.cv.wait_for(lambda:not rt.in_iter and not rt.X.ops,timeout=120):
                log.warning('NestQuant rank %d: engine not idle before graph capture (%d ops in flight)',rt.rank,len(rt.X.ops))
        return b0(g,*a,**k)
    def end(g,*a,**k):
        try:return e0(g,*a,**k)
        finally:
            with rt.cv:rt.ncap-=1;rt.cv.notify_all()
    G.capture_begin,G.capture_end,G._nq_gated=begin,end,True

RT=Runtime()

def forward(L,x,topk_weights,topk_ids):
    d=RT.lay[L];M=d['M'];T=x.shape[0]
    xh=x.half().contiguous();w=topk_weights.half().contiguous();ids=topk_ids.long().contiguous()
    if CHECK and not torch.cuda.is_current_stream_capturing():
        assert x.dim()==2 and ids.shape==(T,TOPK) and w.shape==(T,TOPK),(x.shape,ids.shape,w.shape)
        lo,hi=int(ids.min()),int(ids.max());assert 0<=lo and hi<NE,(L,lo,hi)
        if not d.get('checked'):
            d['checked']=1;t0=d.get('table0');eq=None if t0 is None else bool(torch.equal(t0,M.table))
            allc={k:int((dk['table0']!=dk['M'].table).any(1).sum()) for k,dk in RT.lay.items() if 'table0' in dk}
            log.info('NestQuant pre-call L%d rank %d: changed table rows per layer %s',L,RT.rank,allc)
            ws={k:{n:int((getattr(dk['M'],n)!=0).sum()) for n in ('acc_gu','h','acc_d','cnt_gu','cnt_d','wq')} for k,dk in RT.lay.items()}
            log.info('NestQuant pre-call L%d rank %d: nonzero workspace %s, ptrs acc_d %x table %x',L,RT.rank,ws,M.acc_d.data_ptr(),M.table.data_ptr())
            tot=0.
            for E,e in d['ex'].items():
                for q in (e.gu.base,e.gu.var,e.dn.base,e.dn.var,e.sc[2],e.sc[4]):tot+=float(q.float().sum())
                if e.lr is not None:tot+=float(e.lr.float().sum())
            torch.cuda.synchronize()
            log.info('NestQuant check L%d rank %d: table unchanged %s, resident sum %.4g, dev %s cur %d, M.M %s, H %d I %d, out %s',L,RT.rank,eq,tot,
                     x.device,torch.cuda.current_device(),getattr(M.M,'__file__',M.M),M.H,M.I,tuple(M.out.shape))
        if os.environ.get('NQ_DUMP'):
            torch.save(dict(L=L,x=x.cpu(),w=topk_weights.cpu(),ids=topk_ids.cpu(),table=M.table.cpu(),stride=x.stride(),dev=str(x.device),
                            cur=torch.cuda.current_device(),stream=torch.cuda.current_stream().cuda_stream),f"{os.environ['NQ_DUMP']}/in_r{RT.rank}.pt")
    d['MB'].apply()
    if RT.CAP is not None:RT.CAP.layer(L,x,topk_weights,topk_ids)
    if CHECK and not torch.cuda.is_current_stream_capturing():
        try:torch.cuda.synchronize()
        except Exception as e:raise RuntimeError(f'NestQuant mailbox.apply faulted (L{L})') from e
    sh=d.get('sal_host')
    if T<=BMAX:
        if sh is not None:RT.SAL.sal(x if x.stride(-1)==1 and x.dtype in (torch.bfloat16,torch.float16) else xh,w,ids,d['sal_rsf'],d['sal_acc'],sh.data_ptr())
        return M(xh,ids,w,cfg_gu=CFG_GU[T],cfg_dn=CFG_DN[T]).to(x.dtype)
    if PF and T>=PF_MIN and not torch.cuda.is_current_stream_capturing() and not os.path.exists(PF_OFF):
        if RT.LA is not None:RT.LA.pre(L,x,ids,w,M.table)   # rank 0: router of L+d on x -> level ops (streaming thread)
        return M.prefill(xh,ids,w).to(x.dtype)   # each routed expert decoded once at its live table level + grouped GEMMs
    out=torch.empty(T,x.shape[1],dtype=torch.float32,device=x.device)
    for i in range(0,T,BMAX):
        j=min(T,i+BMAX);M(xh[i:j],ids[i:j],w[i:j],out=out[i:j],cfg_gu=CFG_GU[j-i],cfg_dn=CFG_DN[j-i])
        if sh is not None and T<=16:RT.SAL.sal(xh[i:j],w[i:j],ids[i:j],d['sal_rsf'],d['sal_acc'],sh.data_ptr())   # steps > 16 tok = prefill, ignored by the predictor
        if CHECK and not torch.cuda.is_current_stream_capturing():_check_tables(L,i,xh[i:j],ids[i:j],w[i:j])
    return out.to(x.dtype)

def _check_tables(L,i,x,ids,w):
    torch.cuda.synchronize()
    for k,dk in RT.lay.items():
        t0=dk.get('table0')
        if t0 is None or torch.equal(t0,dk['M'].table):continue
        rows=(t0!=dk['M'].table).any(1).nonzero().flatten().tolist()
        if os.environ.get('NQ_DUMP'):
            torch.save(dict(L=L,i=i,x=x.cpu(),ids=ids.cpu(),w=w.cpu(),k=k,t0=t0.cpu(),t=dk['M'].table.cpu()),f"{os.environ['NQ_DUMP']}/bad_r{RT.rank}.pt")
        raise RuntimeError(f'NestQuant: table of L{k} changed after L{L} chunk {i} (rows {rows[:8]}, n={len(rows)}), '
                           f'thread {threading.current_thread().name} stream {torch.cuda.current_stream().cuda_stream}')

@torch.library.custom_op('nq::moe',mutates_args=())
def nq_moe(x:torch.Tensor,topk_weights:torch.Tensor,topk_ids:torch.Tensor,layer:int)->torch.Tensor:
    return forward(layer,x,topk_weights,topk_ids)
@nq_moe.register_fake
def _(x,topk_weights,topk_ids,layer):return torch.empty_like(x)

def _noop_loader(param,loaded,*a,**k):return None

def make_method(base_cls):
    """NestQuantMoEMethod over the image's ArvqExpertsMoEMethod (keeps its FusedMoE method plumbing)."""
    from vllm.model_executor.utils import set_weight_attrs
    class NestQuantMoEMethod(base_cls):
        def __init__(s,*a,nq_layer,**k):
            super().__init__(*a,**k);s.nq_layer=nq_layer;RT.expect_layer(nq_layer)
        def create_weights(s,layer,num_experts,hidden_size,intermediate_size_per_partition,params_dtype,**extra):
            global RSF
            if RSF is None:
                try:
                    from vllm.config import get_current_vllm_config
                    RSF=float(getattr(get_current_vllm_config().model_config.hf_config,'routed_scaling_factor',1.0) or 1.0)
                except Exception:RSF=None
            assert num_experts==NE and intermediate_size_per_partition*s._tp==2048,(num_experts,intermediate_size_per_partition)
            for n in ARVQ_NAMES:
                p=torch.nn.Parameter(torch.empty(0,dtype=torch.uint8),requires_grad=False)
                layer.register_parameter(n,p);set_weight_attrs(p,{'weight_loader':_noop_loader})
        def process_weights_after_loading(s,layer):
            dev=torch.device('cuda',torch.cuda.current_device())
            RT.add_layer(s.nq_layer,s._tpr,s._tp,dev)
        def apply(s,layer,x,topk_weights,topk_ids,shared_experts=None,shared_experts_input=None):
            act=str(getattr(layer.activation,'value',layer.activation))
            if not act.lower().endswith('silu'):raise ValueError(f'NestQuant requires SiLU, got {act}')
            return torch.ops.nq.moe(x,topk_weights,topk_ids,s.nq_layer)
    return NestQuantMoEMethod
