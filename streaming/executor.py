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
expert simply stays at level 2."""
import torch,numpy as np
import p4rec as PR
from moe import entry

class RankExecutor:
    def __init__(s,rf,layers,nslot,n_host=64,qd=8,device=None,wait_for_slot=True,shadow=False):
        s.shadow=shadow;s.rf=rf;s.lay=rf.lay;s.rb=rf.rb;s.layers=layers
        dev=torch.device('cuda',torch.cuda.current_device() if device is None else device)
        s.slots=torch.empty(nslot,s.rb,dtype=torch.uint8,device=dev);s.free=list(range(nslot))[::-1];s.slot_of={}
        s.eng=rf.engine(n_host,qd,dev.index);s.tag=0;s.ops={}     # tag -> (L, E, kind, seq)
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
            s.ops[s.tag]=(L,E,4,q)
    def poll(s,sched=None,issue=True):
        for tag,hit,trd,te2e in s.eng.poll():
            L,E,kind,q=s.ops.pop(tag)
            if trd<0:                                             # read failed: row never posted
                s.n_failed+=1;s.free.append(s.slot_of.pop((L,E)))
                if sched is not None:sched.failed(L,E,read_error=True)
                continue
            s.layers[L][1].hseq[E]=q;s.wait_apply[L,E]=(kind,q)
            if kind==4:s.lat.append(te2e)
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
    def busy(s):return bool(s.ops or s.wait_apply or s.pend)
    def close(s):s.eng.close()
