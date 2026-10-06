"""Per-rank executor (work item B4): carries out the scheduler's level ops on one TP rank.
One slot pool (P4 records, device) for all layers of the rank, one nqstream engine on that rank's record file, one
thread-13 Mailbox per MoE layer (its apply() is captured at the start of each layer's graph).
  ex = RankExecutor(rankfile, layers={L: (MoELayer, Mailbox, experts{E: kernel expert})}, nslot, n_host, qd)
  ex.apply(ups, downs, sched)   issue ops (no waits); an upgrade with no free slot waits (FIFO) for the next slot a
                                downgrade frees; with wait_for_slot=False it is refused instead (sched.failed)
  ex.poll(sched)                after a replay: ops the mailbox applied -> sched.landed / slot release (issue=False:
                                don't start waiting upgrades, e.g. while a CUDA graph is being captured)
  shadow=True (A/B only): upgrades read + copy into the slot and post a mailbox op, but the posted row is the level-2
                                row, so the served level set never changes (isolates I/O + DMA cost from level-4 share)
Invariants: one outstanding op per expert (the scheduler only downgrades landed experts and only upgrades idle ones);
a slot is reused only after the level-2 row that replaced it was applied; a failed read never posts its row, so the
expert simply stays at level 2.
nq-io extensions (defaults = original behaviour):
  alt_path / qd_alt   dual-drive engine (see nqstream.cu); tier=[(L, E), ...] loads those records into the engine's
                      pinned RAM tier at construction (records whose upgrade then skips the SSD)
  cancel_up(L, E)     best-effort skip of a stale upgrade: dropped if still waiting for a slot here or still queued in
                      the engine (sched.cancelled(L, E) on the poll that reports it); False if it is already reading
  io_stats()          live I/O numbers since the previous call (delivered GB/s, per-drive GB/s / in-flight, queue
                      depths, tier hit share, op latency p50/p99) - the API the throughput-aware scheduler reads
  nd_on=True          nq-tapad (NQ_TAP_ADAPT, set by nq_vllm; off = nothing below happens): land_nd = upgrades that landed
                      without a normal-pool SSD read (RAM-tier / host-LRU hit, prefill-borrow pool slot; the scheduler's
                      landed() takes them out of its drive-rate sample), io_stats drv_GBps = the other landings (drive rate)
nq-prefill (prefill-borrow, sm120/serve/nq_pb.py; nothing changes unless x_borrow() is called):
  x_borrow(ep, addrs) a second, temporary slot pool at device addresses lent by vLLM's KV cache (free blocks) for
                      epoch ep; an upgrade key (L + XB*ep, E) goes to that pool (followers get the same keys from the
                      oplog, so every rank has the same experts there); a stale epoch's key is refused (sched.failed)
  pool_tag(ups)       leader: tag ups for the borrowed pool while it has room and xtag is set
  x_reclaim(sched)    give the pool back: refuse its waiting ups, cancel / wait out every op on its experts, then
                      write their level-2 rows into the mailbox on the current stream (seq bumped, so the next apply of
                      every layer switches them), sync; returns the experts it downgraded (they release normally
                      when applied; their dead slots are never reused)"""
import os,time,collections,torch,numpy as np
XB=65536;XS=1<<20         # nq-prefill: borrowed-pool key L + XB*epoch; borrowed slot id = nslot + XS*epoch + j
import p4rec as PR
def _knob1(p):
    try:return open(p).read().strip()=='1'
    except OSError:return False
from moe import entry

class RankExecutor:
    def __init__(s,rf,layers,nslot,n_host=64,qd=8,device=None,wait_for_slot=True,shadow=False,alt_path=None,qd_alt=0,tier=None):
        s.shadow=shadow;s.rf=rf;s.lay=rf.lay;s.rb=rf.rb;s.layers=layers
        dev=torch.device('cuda',torch.cuda.current_device() if device is None else device)
        s.slots=torch.empty(nslot,s.rb,dtype=torch.uint8,device=dev);s.free=list(range(nslot))[::-1];s.slot_of={}
        s.eng=rf.engine(n_host,qd,dev.index) if not alt_path else rf.engine(n_host,qd,dev.index,alt_path=alt_path,qd_alt=qd_alt)
        s.tag=0;s.ops={}                                          # tag -> (L, E, kind, seq)
        s.up_tag={};s.n_cancel=0;s.n_landed=0;s.lat_w=collections.deque(maxlen=4096);s.lat_rc=collections.deque(maxlen=4096);s._prev={}   # nq-io bookkeeping
        s.tier_n=s.eng.tier_load([rf.rec(L,E) for L,E in tier]) if tier else 0
        s.wait_apply={}                                           # (L, E) -> (kind, seq) device writes done, not yet applied
        s.n_refused=0;s.n_failed=0;s.n_waited=0;s.lat=[];s.wait=wait_for_slot;s.pend=[]
        s.odst={}                                                 # tag -> device destination of an in-flight borrowed-pool upgrade
        s.lent=set()                                              # nq-lmpf slot borrow: normal-pool slots lent to LMPF (never in free)
        s.nd_on=False;s.land_nd=set();s.n_land_d=0;s.prev_d={}       # nq-tapad drive-only landing accounting (nd_on)
        s.nslot=nslot;s.xep=0;s.xfree=[];s.xaddr={};s.xpend=[];s.xtag=False;s.x_have=0;s.x_done=0;s.xst=collections.Counter()   # nq-prefill borrowed pool
        # host-loop cost is GIL time taken from the serving thread: rows are precomputed numpy (level-4 row = template +
        # slot address on the P4 fields), mailbox counters are read through numpy views
        s.slot0=s.slots.data_ptr();s.rc={};s.ah={L:MB.applied_host.numpy() for L,(M,MB,_) in layers.items()}
        s.pp={L:(MB.stage.data_ptr(),MB.stage.stride(0)*8,MB.seq.data_ptr()) for L,(M,MB,_) in layers.items()}   # stage row / seq addresses
        for L,(_,_,exs) in layers.items():
            for E in exs:s.rc[L,E]=s._mkrows(L,E)
    def _mkrows(s,L,E):
        ex=s.layers[L][2][E];sg=ex.signs;ex.signs=ex.sc[2];r2=entry(ex,2);ex.signs=sg
        z=PR.row(ex,s.lay,0,entry);o=PR.row(ex,s.lay,1,entry)
        return r2,z.numpy().copy(),(o-z).numpy().copy()            # (level-2 row, level-4 row at slot 0, slot-address mask)
    def _row(s,L,E,lv,slot=None):
        r2,z,m=s.rc[L,E]
        if lv==2 or s.shadow:return r2            # shadow (A/B only): every read / copy / mailbox op happens, the row stays level 2
        return torch.from_numpy(z+m*(s.slot0+slot*s.rb if slot<s.nslot else s.xaddr[slot]))
    def _rel(s,sl):
        if sl<s.nslot:
            if sl not in s.lent:s.free.append(sl)
        elif s.xep and (sl-s.nslot)//XS==s.xep:s.xfree.append(sl)
        else:s.xaddr.pop(sl,None)                          # borrowed slot of a reclaimed epoch: never reused
    def apply(s,ups,downs,sched=None):
        for L,E in downs:
            MB=s.layers[L][1];st,sb,sq=s.pp[L];s.tag+=1;q=MB.hseq[E]+1
            s.eng.post(s.tag,st+sb*E,s._row(L,E,2),sq+4*E,q);s.ops[s.tag]=(L,E,2,q)
        for L,E in ups:
            fr,pd,k=s.free,s.pend,(L,E)
            if L>=XB:                                       # nq-prefill: upgrade into the borrowed pool of epoch ep
                ep,L=divmod(L,XB)
                if ep!=s.xep:
                    s.xst['stale']+=1
                    if sched is not None:sched.failed(L,E)
                    continue
                fr,pd=s.xfree,s.xpend
            if not fr and s.wait:pd.append(k);s.n_waited+=1;continue
            if not fr:
                s.n_refused+=1
                if sched is not None:sched.failed(L,E)
                continue
            MB=s.layers[L][1];st,sb,sq=s.pp[L];sl=fr.pop();s.slot_of[L,E]=sl;s.tag+=1;q=MB.hseq[E]+1
            dst=s.slot0+sl*s.rb if sl<s.nslot else s.xaddr[sl]
            s.eng.upgrade(s.tag,s.rf.rec(L,E),dst,st+sb*E,s._row(L,E,4,sl),sq+4*E,q)
            s.ops[s.tag]=(L,E,4,q);s.up_tag[L,E]=s.tag
            if s.nd_on:s.land_nd.discard((L,E))
            if sl>=s.nslot:s.odst[s.tag]=dst                   # borrowed-pool destination: x_reclaim fences on it
    def poll(s,sched=None,issue=True):
        for tag,hit,trd,te2e in s.eng.poll():
            L,E,kind,q=s.ops.pop(tag);s.odst.pop(tag,None)
            if kind==4:s.up_tag.pop((L,E),None)
            if trd<=-1e8:                                         # cancelled before it started (cancel_up)
                s.n_cancel+=1;s._rel(s.slot_of.pop((L,E)))
                if sched is not None:(getattr(sched,'cancelled',None) or sched.failed)(L,E)
                continue
            if trd<0:                                             # read failed: row never posted
                s.n_failed+=1;s._rel(s.slot_of.pop((L,E)))
                if sched is not None:sched.failed(L,E,read_error=True)
                continue
            s.layers[L][1].hseq[E]=q;s.wait_apply[L,E]=(kind,q)
            if kind==4:s.lat.append(te2e);s.lat_w.append(te2e);s.lat_rc.append((trd,te2e-trd));s.n_landed+=1
            if kind==4 and s.nd_on:
                if hit or s.slot_of.get((L,E),0)>=s.nslot:s.land_nd.add((L,E))
                else:s.n_land_d+=1
        for (L,E),(kind,q) in list(s.wait_apply.items()):
            if s.ah[L][E]!=q:continue
            del s.wait_apply[L,E]
            if kind==4:
                if sched is not None:sched.landed(L,E)
            else:
                s._rel(s.slot_of.pop((L,E)))
                if sched is not None:sched.released(L,E)
        if issue and s.pend and s.free:
            n=min(len(s.pend),len(s.free));go,s.pend=s.pend[:n],s.pend[n:];s.apply(go,[],sched)
        if issue and s.xpend and s.xfree:
            n=min(len(s.xpend),len(s.xfree));go,s.xpend=s.xpend[:n],s.xpend[n:];s.apply(go,[],sched)
    def cancel_up(s,L,E,sched=None):
        """best effort: drop a not-yet-started upgrade of (L, E). Waiting for a slot here: dropped now (True, callback
        sched.cancelled now). Queued in the engine: cancel requested (True; poll() reports it, or it completes normally
        if it had already started). Otherwise False."""
        k=(L,E)
        xk=next((x for x in s.xpend if x[1]==E and x[0]%XB==L),None) if s.xpend else None
        if xk is not None:s.xpend.remove(xk);s.n_cancel+=1
        if xk is not None or k in s.pend:
            if xk is None:s.pend.remove(k);s.n_cancel+=1
            if sched is not None:(getattr(sched,'cancelled',None) or sched.failed)(L,E)
            return True
        t=s.up_tag.get(k)
        if t is None:return False
        s.eng.cancel(t);return True
    def io_stats(s,name='default'):
        """live I/O numbers over the interval since this caller's (name) previous io_stats() call (first call: since start)"""
        e=s.eng.stats();t=e['now'];c=dict(t=t,landed=s.n_landed,ssd=e['bytes_read'],tier=e['tier_bytes'],hits=e['host_hits'],
                                         th=e['tier_hits'],ups=e['upgrades'],db=list(e['drive_bytes']),dr=list(e['drive_reads']),ds=list(e['drive_read_s']))
        p=s._prev.get(name) or dict(t=t-1e-9,landed=0,ssd=0,tier=0,hits=0,th=0,ups=0,db=[0]*len(c['db']),dr=[0]*len(c['dr']),ds=[0.]*len(c['ds']));s._prev[name]=c
        dt=max(t-p['t'],1e-9);dups=max(c['ups']-p['ups'],0);lat=np.array(s.lat_w) if s.lat_w else None
        ddr=[a-b for a,b in zip(c['dr'],p['dr'])]
        return dict(dt_s=dt,
            delivered_GBps=(c['landed']-p['landed'])*s.rb/dt/1e9,            # upgrades landed in GPU slots
            ssd_GBps=(c['ssd']-p['ssd'])/dt/1e9,
            drive_GBps=[(a-b)/dt/1e9 for a,b in zip(c['db'],p['db'])],
            drive_read_ms=[(a-b)/n*1e3 if n else None for a,b,n in zip(c['ds'],p['ds'],ddr)],   # mean SSD service time per read
            drive_inflight=list(e['drive_inflight']),drive_qd=list(e['drive_qd']),
            tier_GBps=(c['tier']-p['tier'])/dt/1e9,tier_recs=e['tier_recs'],tier_state=e['tier_state'],
            tier_hit_share=(c['th']-p['th'])/dups if dups else None,host_lru_hit_share=(c['hits']-p['hits'])/dups if dups else None,
            eng_waiting=e['waiting'],eng_reading=e['reading'],eng_copying=e['copying'],slot_wait=len(s.pend),ops_outstanding=len(s.ops),
            free_slots=len(s.free),cancelled=s.n_cancel+0,
            ssd_cum=int(c['ssd']),tier_cum=int(c['tier']),landed_cum=int(c['landed']),
            op_p50_ms=float(np.percentile(lat,50))*1e3 if lat is not None else None,
            op_p99_ms=float(np.percentile(lat,99))*1e3 if lat is not None else None,
            **s._rc_stats(),**s._drv(name,t))
    def _drv(s,name,t):    # nq-tapad (nd_on): drive-only landed rate
        if not s.nd_on:return {}
        p=s.prev_d.get(name) or (t-1e-9,0);s.prev_d[name]=(t,s.n_land_d)
        return dict(drv_GBps=(s.n_land_d-p[1])*s.rb/max(t-p[0],1e-9)/1e9)
    def _rc_stats(s):      # nq-kld: op latency split: issue -> read done (engine queue + SSD) and read done -> copy done (ring wait + h2d)
        if not s.lat_rc:return {}
        a=np.array(s.lat_rc)*1e3;return dict(rd_p50_ms=float(np.percentile(a[:,0],50)),rd_p99_ms=float(np.percentile(a[:,0],99)),
                                              cp_p50_ms=float(np.percentile(a[:,1],50)),cp_p99_ms=float(np.percentile(a[:,1],99)))
    def busy(s):return bool(s.ops or s.wait_apply or s.pend or s.xpend)
    # ---- nq-prefill borrowed pool
    def _isx(s,sl):return sl>=s.nslot
    def x_borrow(s,ep,addrs):
        assert not s.xep and ep>s.x_have,(s.xep,ep,s.x_have)
        ids=[s.nslot+XS*ep+j for j in range(len(addrs))]
        s.xaddr.update(zip(ids,addrs));s.xfree=ids[::-1];s.xep=ep;s.x_have=ep;s.xst['borrows']+=1;s.xst['slots']+=len(ids)
    def pool_tag(s,ups):
        if not s.xtag or not ups:return ups
        n=len(s.xfree)-len(s.xpend)
        if n<=0:return ups
        ep=XB*s.xep;s.xst['tagged']+=min(n,len(ups))
        return [(L+ep,E) if j<n else (L,E) for j,(L,E) in enumerate(ups)]
    def x_reclaim(s,sched=None,log=None,warn_s=5.):
        ep=s.xep
        if not ep:return []
        s.xtag=False;t0=time.time()
        for k in s.xpend:
            s.xst['pend_drop']+=1
            if sched is not None:sched.failed(k[0]%XB,k[1])
        s.xpend=[]
        xs=lambda:{k for k,sl in s.slot_of.items() if sl>=s.nslot}
        for k in xs():
            t=s.up_tag.get(k)
            if t is not None and k not in s.wait_apply:
                try:s.eng.cancel(t);s.xst['cancel_req']+=1
                except Exception:pass
        X0=xs();ex=sum(1 for t in s.odst if t in s.ops and (s.ops[t][0],s.ops[t][1]) not in X0)
        if ex:s.xst['fence_extra']+=ex                            # in-flight writes into lent memory the mapping no longer shows
        nw=0;tw=t0;nofence=_knob1('/dev/shm/nq_kvoff_nofence')   # diagnostic (file content 1): the pre-fence (mapping-only) wait
        if nofence:s.xst['nofence']+=1
        while True:                                       # every op on a borrowed-slot expert must have written
            X=xs()
            # by mapping (ops of experts in borrowed slots) AND by destination (any op still writing into lent memory,
            # whatever the expert -> slot mapping says now)
            if not any((o[0],o[1]) in X for o in s.ops.values()) and (nofence or not any(t in s.ops for t in s.odst)):break
            s.poll(sched,issue=False);time.sleep(2e-4);nw+=1
            if time.time()-tw>warn_s:
                tw=time.time()
                if log is not None:log.warning('NestQuant prefill-borrow: reclaim of epoch %d still waiting for %d ops (%.1fs)',ep,
                                               sum((o[0],o[1]) in X for o in s.ops.values()),tw-t0)
        forced=[];by=collections.defaultdict(list)
        for (L,E),sl in s.slot_of.items():
            if sl<s.nslot:continue
            w=s.wait_apply.get((L,E))
            if w is not None and w[0]==2:continue          # its level-2 row is already staged
            by[L].append(E)
        dev=s.slots.device
        for L,E,qq in s._stage({L:[(E,s.rc[L,E][0]) for E in Es] for L,Es in by.items()}):s.wait_apply[L,E]=(2,qq);forced.append((L,E))
        torch.cuda.synchronize(dev)
        if s.odst:s.xst['odst_stale']+=len(s.odst);s.odst={k:v for k,v in s.odst.items() if k in s.ops}
        s.xep=0;s.xfree=[];s.x_done=ep
        for sl in [sl for sl in s.xaddr if sl not in set(s.slot_of.values())]:del s.xaddr[sl]
        s.xst['forced']+=len(forced);s.xst['reclaims']+=1;s.xst['wait_iters']+=nw;s.xst['reclaim_ms']+=int((time.time()-t0)*1e3)
        return forced
    # ---- staged rows (x_reclaim, nq-lmpf slot borrow)
    def _stage(s,by):
        """by = {L: [(E, row)]}: write the rows into the mailbox stage + bump seq on the current stream -> [(L, E, seq)]
        (the next apply of the layer, captured or direct, switches the table rows)"""
        dev=s.slots.device;out=[]
        for L,er in by.items():
            if not er:continue
            MB=s.layers[L][1];Es=[E for E,_ in er];q=[MB.hseq[E]+1 for E in Es]
            rows=torch.stack([torch.as_tensor(r) for _,r in er]).to(device=dev,dtype=MB.stage.dtype)
            ix=torch.tensor(Es,dtype=torch.long,device=dev)
            MB.stage.index_copy_(0,ix,rows);MB.seq.index_copy_(0,ix,torch.tensor(q,dtype=MB.seq.dtype,device=dev))
            for E,qq in zip(Es,q):MB.hseq[E]=qq;out.append((L,E,qq))
        return out
    def _stage_apply(s,by):
        """stage, apply the touched layers' mailboxes now, sync; the table rows have switched when this returns"""
        st=s._stage(by)
        for L in {L for L,_,_ in st}:s.layers[L][1].apply()
        torch.cuda.synchronize(s.slots.device)
        bad=[(L,E) for L,E,q in st if int(s.ah[L][E])!=q]
        assert not bad,f'mailbox apply did not land {bad[:4]}'
        return st
    def quiet(s):
        """no op in flight, nothing waiting for a mailbox apply, no borrowed (prefill-borrow) pool"""
        return not s.ops and not s.wait_apply and not s.xep and not any(sl>=s.nslot for sl in s.slot_of.values())
    def evict_now(s,keys):
        """level-4 residents keys -> level 2 now (rows applied, slots released to free unless lent). Executor must be
        quiet (no ops / applies outstanding). -> the keys evicted (callers update their scheduler / follower)"""
        ks=[k for k in keys if k in s.slot_of and s.slot_of[k]<s.nslot]
        by=collections.defaultdict(list)
        for L,E in ks:by[L].append((E,s.rc[L,E][0]))
        if ks:s._stage_apply(by)
        for k in ks:s._rel(s.slot_of.pop(k))
        s.xst['lend_evict']+=len(ks);return ks
    def relocate(s,moves):
        """[(L, E), src, dst]: copy the record bytes src -> dst (device), switch the level-4 row to dst, src freed"""
        if not moves:return
        fr=set(s.free)
        for k,a,b in moves:
            assert s.slot_of.get(k)==a and b in fr and b not in s.lent,(k,a,b)
            s.slots[b].copy_(s.slots[a]);fr.discard(b)
        by=collections.defaultdict(list)
        for (L,E),a,b in moves:by[L].append((E,s._row(L,E,4,b)))
        s._stage_apply(by)
        dst={b for _,_,b in moves};s.free=[x for x in s.free if x not in dst]
        for k,a,b in moves:s.slot_of[k]=b;s._rel(a)
        s.xst['lend_move']+=len(moves)
    def lend(s,a,n):
        """slots [a, a+n) leave the pool (must hold no expert): removed from free, never released into it until unlend"""
        sp=set(range(a,a+n));assert a>=0 and a+n<=s.nslot and not (sp&s.lent)
        occ=[k for k,sl in s.slot_of.items() if sl in sp];assert not occ,f'lend: span holds {occ[:4]}'
        s.free=[x for x in s.free if x not in sp];s.lent|=sp;s.xst['lends']+=1;s.xst['lent']+=n
    def unlend(s):
        n=len(s.lent);s.free.extend(sorted(s.lent,reverse=True));s.lent=set();return n
    def addr(s,sl):return s.slot0+sl*s.rb
    def span_view(s,a,n):
        """uint8 view of slots [a, a+n) (contiguous bytes at addr(a))"""
        return s.slots[a:a+n].view(-1)
    def close(s):s.eng.close()
