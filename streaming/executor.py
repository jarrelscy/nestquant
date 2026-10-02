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
                      depths, tier hit share, op latency p50/p99) - the API the throughput-aware scheduler reads"""
import time,collections,torch,numpy as np
import p4rec as PR
from moe import entry

class RankExecutor:
    def __init__(s,rf,layers,nslot,n_host=64,qd=8,device=None,wait_for_slot=True,shadow=False,alt_path=None,qd_alt=0,tier=None):
        s.shadow=shadow;s.rf=rf;s.lay=rf.lay;s.rb=rf.rb;s.layers=layers
        dev=torch.device('cuda',torch.cuda.current_device() if device is None else device)
        s.slots=torch.empty(nslot,s.rb,dtype=torch.uint8,device=dev);s.free=list(range(nslot))[::-1];s.slot_of={}
        s.eng=rf.engine(n_host,qd,dev.index) if not alt_path else rf.engine(n_host,qd,dev.index,alt_path=alt_path,qd_alt=qd_alt)
        s.tag=0;s.ops={}                                          # tag -> (L, E, kind, seq)
        s.up_tag={};s.n_cancel=0;s.n_landed=0;s.lat_w=collections.deque(maxlen=4096);s._prev={}   # nq-io bookkeeping
        s.tier_n=s.eng.tier_load([rf.rec(L,E) for L,E in tier]) if tier else 0
        s.wait_apply={}                                           # (L, E) -> (kind, seq) device writes done, not yet applied
        s.n_refused=0;s.n_failed=0;s.n_waited=0;s.lat=[];s.wait=wait_for_slot;s.pend=[]
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
        return torch.from_numpy(z+m*(s.slot0+slot*s.rb))
    def apply(s,ups,downs,sched=None):
        for L,E in downs:
            MB=s.layers[L][1];st,sb,sq=s.pp[L];s.tag+=1;q=MB.hseq[E]+1
            s.eng.post(s.tag,st+sb*E,s._row(L,E,2),sq+4*E,q);s.ops[s.tag]=(L,E,2,q)
        for L,E in ups:
            if not s.free and s.wait:s.pend.append((L,E));s.n_waited+=1;continue
            if not s.free:
                s.n_refused+=1
                if sched is not None:sched.failed(L,E)
                continue
            MB=s.layers[L][1];st,sb,sq=s.pp[L];sl=s.free.pop();s.slot_of[L,E]=sl;s.tag+=1;q=MB.hseq[E]+1
            s.eng.upgrade(s.tag,s.rf.rec(L,E),s.slot0+sl*s.rb,st+sb*E,s._row(L,E,4,sl),sq+4*E,q)
            s.ops[s.tag]=(L,E,4,q);s.up_tag[L,E]=s.tag
    def poll(s,sched=None,issue=True):
        for tag,hit,trd,te2e in s.eng.poll():
            L,E,kind,q=s.ops.pop(tag)
            if kind==4:s.up_tag.pop((L,E),None)
            if trd<=-1e8:                                         # cancelled before it started (cancel_up)
                s.n_cancel+=1;s.free.append(s.slot_of.pop((L,E)))
                if sched is not None:(getattr(sched,'cancelled',None) or sched.failed)(L,E)
                continue
            if trd<0:                                             # read failed: row never posted
                s.n_failed+=1;s.free.append(s.slot_of.pop((L,E)))
                if sched is not None:sched.failed(L,E,read_error=True)
                continue
            s.layers[L][1].hseq[E]=q;s.wait_apply[L,E]=(kind,q)
            if kind==4:s.lat.append(te2e);s.lat_w.append(te2e);s.n_landed+=1
        for (L,E),(kind,q) in list(s.wait_apply.items()):
            if s.ah[L][E]!=q:continue
            del s.wait_apply[L,E]
            if kind==4:
                if sched is not None:sched.landed(L,E)
            else:
                s.free.append(s.slot_of.pop((L,E)))
                if sched is not None:sched.released(L,E)
        if issue and s.pend and s.free:
            n=min(len(s.pend),len(s.free));go,s.pend=s.pend[:n],s.pend[n:];s.apply(go,[],sched)
    def cancel_up(s,L,E,sched=None):
        """best effort: drop a not-yet-started upgrade of (L, E). Waiting for a slot here: dropped now (True, callback
        sched.cancelled now). Queued in the engine: cancel requested (True; poll() reports it, or it completes normally
        if it had already started). Otherwise False."""
        k=(L,E)
        if k in s.pend:
            s.pend.remove(k);s.n_cancel+=1
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
            op_p50_ms=float(np.percentile(lat,50))*1e3 if lat is not None else None,
            op_p99_ms=float(np.percentile(lat,99))*1e3 if lat is not None else None)
    def busy(s):return bool(s.ops or s.wait_apply or s.pend)
    def close(s):s.eng.close()
