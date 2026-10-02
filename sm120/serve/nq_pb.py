"""nq-prefill: prefill-borrow, worker side (every TP rank; NQ_PREFILL_BORROW=1, see nq_pb_engine.py for the engine side).

The scheduler hands free KV blocks to the expert slot pool during a long prefill and states it on every
SchedulerOutput (nq_pb = (epoch, phase, runs, n_slots) or None). Worker.execute_model is wrapped; before the step's
forward (so before its KV writes and LMCache loads) and with the NQ streaming loop paused:
  new epoch  synchronize (no earlier step still touches those blocks), carve n_slots rb-sized slots out of the KV
             storages over the block runs (same rule on every rank: per run and storage the start is aligned up to
             ALIGN and floor((run bytes - ALIGN) / rb) slots are cut), X.x_borrow(epoch, addrs).
             Leader: S.slots += n, S.nf = NQ_PREFILL_SLOTS (lookahead now plans that many per layer), decode residents
             D0 snapshotted, X.xtag on while phase == 'borrow' (ups go to the borrowed pool while it has room).
  no epoch   (or a different one) X.x_reclaim(): every op on a borrowed-slot expert finished or cancelled, their level-2
             rows staged in the mailbox (each layer switches at its next apply, before its MoE reads anything), device
             synchronized -> the blocks may be written from this step on.
             Leader: those experts -> state 3 (released when applied), S.slots / S.nf back, RESTORE: want = D0 (unless a
             session-restore pin is active), reclaim marker into the oplog. Followers: same reclaim on their own pool
             (OpLog gate keeps it at the leader's point of the op order).
Decode (no borrow) is untouched: nothing here runs unless nq_pb was set on some output. Any error -> borrowing off for
the boot (the pool is reclaimed first).

Phase 2 (nq_pb[4] = kv dict, NQ_PREFILL_KV_OFFLOAD=1), class KVOff:
  begin   synchronize; the request's rows (its MLA-group blocks) of the first n_off MLA layers' KV are copied to a pinned
          host buffer (cudaHostRegister of exactly n_off x rows x page bytes); every offloaded layer's attn.kv_cache now
          points at one of two staging buffers (the original storages of offloaded layers 0 and 1); the storages of
          offloaded layers 2.. are carved into expert slots. LMCache's engine.store is deferred (its registered tensors
          are the original storages).
  step    (worker hook, before the forward) write-back of the previous step's last layer, new rows, prefetch layers 0/1.
  touch   mla_attention.get_attention_context is wrapped: at the first call for offloaded layer k in a step (its KV
          update, before any of its kernels): on a side stream, after an event on the compute stream, the blocks the
          step wrote in the layer before go back to host, layer k+1 is prefetched into the other buffer; the compute
          stream waits for layer k's H2D; attn.kv_cache = layer k's buffer.
  end     (after X.x_reclaim: no expert lives there any more) write-back, synchronize, every row back into its original
          storage, original tensors re-bound (decode cudagraphs keep their pointers), host unregistered and freed, the
          deferred LMCache stores replayed in order (same kwargs: the slots hold the same KV again), synchronize."""
import os,time,functools,logging,collections
import numpy as np,torch
try:
    from vllm.logger import init_logger;log=init_logger('vllm.nestquant.pb')
except Exception:log=logging.getLogger('nestquant.pb')
from nq_pb_engine import ON,PF_NF,ALIGN,carve_count,KV_OFF,mla_candidate,layer_idx

def kv_storages(runner,nb):
    """[(base address, page bytes)] of the distinct KV storages whose block b is bytes [b*page, (b+1)*page) of the
    storage for every view on it (others are skipped: never carved)"""
    if getattr(runner,'cross_layers_kv_cache',None) is not None:return [],'cross-layer KV layout'
    seen={};bad=set();ts=[]
    def walk(x):
        if isinstance(x,torch.Tensor):ts.append(x)
        elif isinstance(x,(list,tuple)):
            for y in x:walk(y)
    walk(getattr(runner,'kv_caches',[]))
    for t in ts:
        st=t.untyped_storage();p0=st.data_ptr();n=st.nbytes()
        if n%nb:bad.add(p0);continue
        pg=n//nb;seen[p0]=pg
        # block-major: some dim of size nb has the largest stride and stride*itemsize == page
        ok=any(t.shape[d]==nb and t.stride(d)*t.element_size()==pg and t.stride(d)==max(t.stride()) for d in range(t.dim()))
        if not ok or t.storage_offset()*t.element_size()>=pg:bad.add(p0)
    out=[(p,pg) for p,pg in seen.items() if p not in bad]
    return sorted(out),(f'{len(bad)} storages not block-major' if bad else '')

def carve(stor,runs,rb,want):
    """slot addresses over the runs (in order), per run and storage exactly carve_count's number"""
    out=[]
    for b0,b1 in runs:
        for base,pg in stor:
            n=carve_count(b1-b0,[pg],rb)
            a=base+b0*pg;a=(a+ALIGN-1)//ALIGN*ALIGN
            assert a+n*rb<=base+b1*pg
            out.extend(a+j*rb for j in range(n))
            if len(out)>=want:return out[:want]
    return out

class KVOff:
    """phase 2 on one rank (see the module doc)"""
    def __init__(s,rt):
        s.rt=rt;s.on=False;s.dead=False;s.cand=None;s.cs=None;s.n=collections.Counter();s.lm=None;s.deferred=[]
    @staticmethod
    def _kv(L):
        kv=getattr(L,'kv_cache',None)
        if isinstance(kv,(list,tuple)):kv=kv[0] if kv else None
        return kv if isinstance(kv,torch.Tensor) and kv.numel() else None
    @staticmethod
    def _setkv(L,t):
        if isinstance(L.kv_cache,list):L.kv_cache[0]=t
        else:L.kv_cache=t
    def candidates(s,runner):
        ctx=runner.compilation_config.static_forward_context;nb=runner.kv_cache_config.num_blocks
        nL=int(runner.model_config.hf_text_config.num_hidden_layers)
        users=collections.Counter()
        for L in ctx.values():
            kv=s._kv(L)
            if kv is not None:users[kv.untyped_storage().data_ptr()]+=1
        out=[]
        for name,L in ctx.items():
            if not mla_candidate(name,nL) or type(L).__name__!='MLAAttention':continue
            kv=s._kv(L)
            if kv is None:continue
            st=kv.untyped_storage();n=st.nbytes()
            if users[st.data_ptr()]!=1 or n%nb or kv.storage_offset() or kv.shape[0]!=nb or not kv.is_contiguous():continue
            pg=n//nb
            if kv.stride(0)*kv.element_size()!=pg:continue
            flat=torch.empty(0,dtype=torch.uint8,device=kv.device).set_(st,0,(nb,pg),(pg,1))
            out.append((layer_idx(name),name,L,kv,flat,pg))
        out.sort(key=lambda x:x[0]);return out
    @staticmethod
    def _runs(rows,blocks):
        """[(r0, r1, b0)] maximal runs where row and block both step by 1"""
        out=[]
        for r,b in zip(rows,blocks):
            if out and out[-1][1]==r and out[-1][2]+(r-out[-1][0])==b:out[-1][1]=r+1
            else:out.append([r,r+1,b])
        return [tuple(x) for x in out]
    def _set_blocks(s,blocks):
        for b in blocks:
            if b not in s.row_of:
                if len(s.order)>=s.rows:raise RuntimeError(f'KV offload: {len(s.order)+1} blocks > {s.rows} host rows')
                s.row_of[b]=len(s.order);s.order.append(b)
        if len(blocks)!=s.nrun_rows:
            s.all_runs=s._runs(range(len(s.order)),s.order);s.nrun_rows=len(blocks)
    def _host_alloc(s,n):
        buf=torch.empty(n,dtype=torch.uint8)
        if buf.device.type=='cpu' and torch.cuda.is_available():
            e=torch.cuda.cudart().cudaHostRegister(buf.data_ptr(),n,0)
            if int(e)!=0:raise RuntimeError(f'cudaHostRegister({n/2**30:.1f} GiB) failed: {e}')
            s.registered=True
        return buf
    def _host_free(s):
        if s.host is not None and getattr(s,'registered',False):torch.cuda.cudart().cudaHostUnregister(s.host_raw.data_ptr())
        s.registered=False;s.host=None;s.host_raw=None
    def _lmcache(s):
        try:
            from vllm.distributed.kv_transfer import has_kv_transfer_group,get_kv_transfer_group
            if not has_kv_transfer_group():return None,True
            g=get_kv_transfer_group();impl=getattr(g,'_lmcache_engine',None);eng=getattr(impl,'lmcache_engine',None)
            if eng is None or not hasattr(eng,'store'):return None,False
            return eng,True
        except Exception:return None,False
    def begin(s,runner,kv,rb,want):
        """-> slot addresses (empty: phase 2 not done this epoch, nothing changed)"""
        if s.dead:return []
        rt=s.rt;dev=rt.dev
        if s.cand is None:
            s.cand=s.candidates(runner)
            log.info('NestQuant prefill-borrow rank %d: KV offload candidates %d MLA layers (pages %s)',rt.rank,len(s.cand),sorted({c[5] for c in s.cand}))
        if len(s.cand)!=kv['n_cand']:
            log.warning('NestQuant prefill-borrow rank %d: %d MLA candidates here, engine says %d: phase 2 off',rt.rank,len(s.cand),kv['n_cand'])
            s.dead=True;return []
        eng,ok=s._lmcache()
        if not ok:log.warning('NestQuant prefill-borrow rank %d: KV connector without a reachable LMCache engine.store: phase 2 off',rt.rank);s.dead=True;return []
        n_off=kv['n_off'];s.off=s.cand[:n_off];s.rows=kv['rows'];s.page=s.off[0][5];s.nb=runner.kv_cache_config.num_blocks;t0=time.time()
        s.host=None;s.host_raw=None
        try:
            torch.cuda.synchronize(dev)
            s.host_raw=s._host_alloc(n_off*s.rows*s.page);s.host=s.host_raw.view(n_off,s.rows,s.page)
            s.row_of={};s.order=[];s.nrun_rows=-1;s._set_blocks(kv['blocks'])
            for k,c in enumerate(s.off):
                for r0,r1,b0 in s.all_runs:s.host[k,r0:r1].copy_(c[4][b0:b0+r1-r0])
            torch.cuda.synchronize(dev)
        except Exception:
            log.exception('NestQuant prefill-borrow rank %d: KV offload begin failed, phase 2 off',rt.rank)
            s._host_free();s.dead=True;return []
        try:
            if s.cs is None:s.cs=torch.cuda.Stream(device=dev)
            np_=patch_gac()
        except Exception:
            log.exception('NestQuant prefill-borrow rank %d: get_attention_context wrap failed, phase 2 off',rt.rank)
            s._host_free();s.dead=True;return []
        if np_:log.info('NestQuant prefill-borrow rank %d: get_attention_context wrapped in %d more modules',rt.rank,np_)
        s.buf=[s.off[0][3],s.off[1][3]];s.bflat=[s.off[0][4],s.off[1][4]];s.holds=[-1,-1];s.hev=[None,None]
        s.ord={c[1]:k for k,c in enumerate(s.off)};s.cur=-1;s.wb_runs=();s.pend=None
        for c in s.off:s._setkv(c[2],s.buf[0])
        s.lm=eng;s.deferred=[]
        if eng is not None:
            d=s.deferred
            def store(*a,**k):d.append((a,k))
            eng.store=store
        addrs=carve([(c[4].data_ptr(),s.page) for c in s.off[2:]],[(0,s.nb)],rb,want)
        s.on=True;s.n['begins']+=1
        log.info('NestQuant prefill-borrow rank %d: KV offload of %d layers, %d rows (%.2f GiB host), %d slots, %.0f ms',rt.rank,n_off,
                 len(s.order),n_off*s.rows*s.page/2**30,len(addrs),(time.time()-t0)*1e3)
        return addrs
    def _wb(s):
        """write-back of the blocks the step wrote in the current layer (after the compute stream's work so far)"""
        if s.cur<0:return
        k=s.cur;b=s.holds.index(k);ev=torch.cuda.Event();ev.record(torch.cuda.current_stream(s.rt.dev))
        with torch.cuda.stream(s.cs):
            s.cs.wait_event(ev)
            for r0,r1,b0 in s.wb_runs:s.host[k,r0:r1].copy_(s.bflat[b][b0:b0+r1-r0],non_blocking=True)
        s.cur=-1
    def _load(s,k,b):
        with torch.cuda.stream(s.cs):
            for r0,r1,b0 in s.all_runs:s.bflat[b][b0:b0+r1-r0].copy_(s.host[k,r0:r1],non_blocking=True)
            ev=torch.cuda.Event();ev.record(s.cs)
        s.holds[b]=k;s.hev[b]=ev;s.n['h2d']+=1
    def step(s,kv):
        """worker hook of every step of the epoch, before the forward"""
        if not s.on:return
        s._wb()
        ev=torch.cuda.Event();ev.record(torch.cuda.current_stream(s.rt.dev));s.cs.wait_event(ev)  # buffers free of the last step
        s._set_blocks(kv['blocks'])
        s.wb_runs=s._runs([s.row_of[b] for b in kv['wb']],kv['wb'])
        s.holds=[-1,-1];s._load(0,0)
        if len(s.off)>1:s._load(1,1)
        s.n['steps']+=1
    def touch(s,name):
        k=s.ord.get(name)
        if k is None or k==s.cur:return
        s._wb()
        if k in s.holds:b=s.holds.index(k)
        else:
            b=0 if s.holds[1]==k+1 else 1;s._load(k,b);s.n['h2d_miss']+=1
        o=1-b
        if k+1<len(s.off) and s.holds[o]!=k+1:s._load(k+1,o)
        torch.cuda.current_stream(s.rt.dev).wait_event(s.hev[b])
        s._setkv(s.off[k][2],s.buf[b]);s.cur=k
    def end(s):
        """after X.x_reclaim"""
        if not s.on:return
        rt=s.rt;t0=time.time();nd=len(s.deferred)
        try:
            s._wb();torch.cuda.synchronize(rt.dev)
            with torch.cuda.stream(s.cs):
                for k,c in enumerate(s.off):
                    for r0,r1,b0 in s.all_runs:c[4][b0:b0+r1-r0].copy_(s.host[k,r0:r1],non_blocking=True)
            torch.cuda.synchronize(rt.dev)
        finally:
            for c in s.off:s._setkv(c[2],c[3])
            s.on=False;s._host_free()
            if s.lm is not None:
                try:del s.lm.store
                except AttributeError:pass
        t1=time.time()
        for a,k in s.deferred:s.lm.store(*a,**k)
        s.deferred=[];s.lm=None
        if nd:torch.cuda.synchronize(rt.dev)
        log.info('NestQuant prefill-borrow rank %d: KV offload ended, %d rows restored in %.0f ms, %d deferred LMCache stores in %.0f ms, %s',
                 rt.rank,len(s.order),(t1-t0)*1e3,nd,(time.time()-t1)*1e3,dict(s.n))

class PB:
    def __init__(s,rt):
        s.rt=rt;s.ep=0;s.dead=False;s.lead=rt.F is None;s.nf0=rt.S.nf;s.extra=0;s.D0=None;s.stor=None;s.n=collections.Counter()
        s.KO=KVOff(rt) if KV_OFF else None
        s.last_log=0.
    # --- streaming loop pause (the loop runs X / S / F; we mutate them from the worker thread)
    def _pause(s):
        rt=s.rt
        with rt.cv:
            rt.pb_pause+=1
            if not rt.cv.wait_for(lambda:not rt.in_iter,timeout=60):log.warning('NestQuant prefill-borrow: streaming loop busy > 60 s')
    def _resume(s):
        rt=s.rt
        with rt.cv:rt.pb_pause-=1;rt.cv.notify_all()
    def on_sched(s,runner,so):
        pb=getattr(so,'nq_pb',None)
        if s.dead:return
        if pb is None and not s.ep:return
        if pb is not None and pb[0]==s.ep:                 # same epoch: phase only (+ phase 2 step)
            if s.lead:s.rt.X.xtag=pb[1]=='borrow' and bool(s.extra)
            if s.KO is not None and s.KO.on and len(pb)>4 and pb[4]:
                try:s.KO.step(pb[4])
                except Exception:
                    log.exception('NestQuant prefill-borrow: KV offload step failed, borrowing off for this boot');s.dead=True
                    s._pause()
                    try:s._reclaim()
                    except Exception:log.exception('NestQuant prefill-borrow: reclaim after error failed')
                    finally:s._resume()
            return
        s._pause()
        try:
            if s.ep:s._reclaim()
            if pb is not None:s._borrow(runner,*pb[:4],kv=pb[4] if len(pb)>4 else None)
        except Exception:
            log.exception('NestQuant prefill-borrow: worker hook failed, borrowing off for this boot')
            s.dead=True
            try:
                if s.ep:s._reclaim()
            except Exception:log.exception('NestQuant prefill-borrow: reclaim after error failed')
        finally:s._resume()
    def _borrow(s,runner,ep,phase,runs,nslots,kv=None):
        rt=s.rt;X=rt.X;t0=time.time()
        torch.cuda.synchronize(rt.dev)
        if kv is not None:
            addrs=s.KO.begin(runner,kv,X.rb,nslots) if s.KO is not None else []
            if s.KO is not None and s.KO.on:s.KO.step(kv)
        elif s.stor is None:
            nb=runner.kv_cache_config.num_blocks;s.stor,why=kv_storages(runner,nb)
            log.info('NestQuant prefill-borrow rank %d: %d KV storages carvable (pages %s)%s',rt.rank,len(s.stor),
                     sorted({p for _,p in s.stor}),f', {why}' if why else '')
        if kv is None:addrs=carve(s.stor,runs,X.rb,nslots) if s.stor else []
        X.x_borrow(ep,addrs);s.ep=ep;s.extra=len(addrs)
        if s.lead:
            S=rt.S;s.D0=np.isin(S.state,(1,2))&~S.fixed
            if S.slots is not None:S.slots+=s.extra
            if s.extra:S.nf=PF_NF
            S.pb_lazy=bool(s.extra);X.xtag=phase=='borrow' and bool(s.extra)
        s.n['borrows']+=1;s.n['slots']+=len(addrs)
        log.info('NestQuant prefill-borrow rank %d: epoch %d borrowed %d slots (%.1f GiB, asked %d) in %d runs, %.1f ms',rt.rank,ep,
                 len(addrs),len(addrs)*X.rb/2**30,nslots,len(runs),(time.time()-t0)*1e3)
    def _reclaim(s):
        rt=s.rt;X=rt.X;ep=s.ep;t0=time.time()
        if s.lead:
            S=rt.S;forced=X.x_reclaim(S,log)
            for L,E in forced:
                i=S.li[L]
                if S.state[i,E] in (1,2):S.state[i,E]=3
            if S.slots is not None:S.slots-=s.extra
            S.nf=s.nf0;S.pb_lazy=False
            if s.D0 is not None and getattr(S,'pin',None) is None:S.want=s.D0.copy()      # RESTORE
            if rt.log is not None:rt.log.put_reclaim(ep)
        else:
            F=rt.F
            if F.chk is not None:                                 # debug check: every borrowed-pool expert leaves the leader's set
                xk={k for k,sl in X.slot_of.items() if sl>=X.nslot}|{(L%65536,E) for L,E in X.xpend}
                q=F.q.items() if isinstance(F.q,dict) else F.q
                xk|={(k[0]%65536,k[1]) for k,lv in list(q) if k[0]>=65536 or lv>4}
            forced=X.x_reclaim(F,log);F.busy.update(forced)
            if F.chk is not None:F.chk.difference_update(xk)
        if s.KO is not None and s.KO.on:s.KO.end()               # after x_reclaim: no expert in those storages
        s.n['reclaims']+=1;s.n['forced']+=len(forced)
        log.info('NestQuant prefill-borrow rank %d: epoch %d reclaimed, %d experts back to level 2, %.1f ms (executor %s)',rt.rank,ep,
                 len(forced),(time.time()-t0)*1e3,dict(X.xst))
        s.ep=0;s.extra=0;s.D0=None

_GAC=[None,None]
def patch_gac(rt=None):
    """every module that bound vllm's get_attention_context (mla_attention, fusion passes) gets the touching wrapper"""
    import sys
    if _GAC[0] is None:
        from vllm.model_executor.layers.attention import attention as A
        g0=A.get_attention_context;_GAC[0]=g0
        def get_attention_context(layer_name,_g0=g0):
            pb=getattr(_GAC[1],'PB',None);ko=pb.KO if pb is not None else None
            if ko is not None and ko.on:ko.touch(layer_name)
            return _g0(layer_name)
        get_attention_context._nq_pb=True;_GAC.append(get_attention_context)
    if rt is not None:_GAC[1]=rt
    n=0
    for m in list(sys.modules.values()):
        try:
            if getattr(m,'get_attention_context',None) is _GAC[0]:setattr(m,'get_attention_context',_GAC[2]);n+=1
        except Exception:pass
    return n

def install(rt):
    """called from Runtime.start on every rank (streaming on, NQ_PREFILL_BORROW=1)"""
    try:from vllm.v1.worker import gpu_worker as GW
    except Exception as e:log.warning('NestQuant prefill-borrow: no gpu_worker (%s), off',e);return None
    rt.pb_pause=0;rt.PB=PB(rt)
    if rt.F is not None and hasattr(rt.F,'gate'):rt.F.gate=rt.X
    W=GW.Worker
    if getattr(W,'_nq_pb',False):return rt.PB
    ex0=W.execute_model
    def execute_model(self,scheduler_output,*a,**k):
        pb=getattr(rt,'PB',None)
        if pb is not None and scheduler_output is not None:
            try:pb.on_sched(self.model_runner,scheduler_output)
            except Exception:log.exception('NestQuant prefill-borrow: hook failed');pb.dead=True
        return ex0(self,scheduler_output,*a,**k)
    functools.update_wrapper(execute_model,ex0);W.execute_model=execute_model;W._nq_pb=True
    if KV_OFF:patch_gac(rt)
    log.info('NestQuant prefill-borrow rank %d: worker hook installed (%s, %d prefill slots/layer)',rt.rank,'leader' if rt.F is None else 'follower',PF_NF)
    return rt.PB
