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
the boot (the pool is reclaimed first)."""
import os,time,functools,logging,collections
import numpy as np,torch
try:
    from vllm.logger import init_logger;log=init_logger('vllm.nestquant.pb')
except Exception:log=logging.getLogger('nestquant.pb')
from nq_pb_engine import ON,PF_NF,ALIGN,carve_count

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

class PB:
    def __init__(s,rt):
        s.rt=rt;s.ep=0;s.dead=False;s.lead=rt.F is None;s.nf0=rt.S.nf;s.extra=0;s.D0=None;s.stor=None;s.n=collections.Counter()
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
        if pb is not None and pb[0]==s.ep:                 # same epoch: phase only
            if s.lead:s.rt.X.xtag=pb[1]=='borrow' and bool(s.extra)
            return
        s._pause()
        try:
            if s.ep:s._reclaim()
            if pb is not None:s._borrow(runner,*pb)
        except Exception:
            log.exception('NestQuant prefill-borrow: worker hook failed, borrowing off for this boot')
            s.dead=True
            try:
                if s.ep:s._reclaim()
            except Exception:log.exception('NestQuant prefill-borrow: reclaim after error failed')
        finally:s._resume()
    def _borrow(s,runner,ep,phase,runs,nslots):
        rt=s.rt;X=rt.X;t0=time.time()
        torch.cuda.synchronize(rt.dev)
        if s.stor is None:
            nb=runner.kv_cache_config.num_blocks;s.stor,why=kv_storages(runner,nb)
            log.info('NestQuant prefill-borrow rank %d: %d KV storages carvable (pages %s)%s',rt.rank,len(s.stor),
                     sorted({p for _,p in s.stor}),f', {why}' if why else '')
        addrs=carve(s.stor,runs,X.rb,nslots) if s.stor else []
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
        s.n['reclaims']+=1;s.n['forced']+=len(forced)
        log.info('NestQuant prefill-borrow rank %d: epoch %d reclaimed, %d experts back to level 2, %.1f ms (executor %s)',rt.rank,ep,
                 len(forced),(time.time()-t0)*1e3,dict(X.xst))
        s.ep=0;s.extra=0;s.D0=None

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
    log.info('NestQuant prefill-borrow rank %d: worker hook installed (%s, %d prefill slots/layer)',rt.rank,'leader' if rt.F is None else 'follower',PF_NF)
    return rt.PB
