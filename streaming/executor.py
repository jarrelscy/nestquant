"""Per-rank executor (work item B4): carries out the scheduler's level ops on one TP rank.
One slot pool (P4 records, device) for all layers of the rank, one nqstream engine on that rank's record file, one
thread-13 Mailbox per MoE layer (its apply() is captured at the start of each layer's graph).
  ex = RankExecutor(rankfile, layers={L: (MoELayer, Mailbox, experts{E: kernel expert})}, nslot, n_host, qd)
  ex.apply(ups, downs, sched)   issue ops (no waits); an upgrade with no free slot waits (FIFO) for the next slot a
                                downgrade frees; with wait_for_slot=False it is refused instead (sched.failed)
  ex.poll(sched)                after a replay: ops the mailbox applied -> sched.landed / slot release
Invariants: one outstanding op per expert (the scheduler only downgrades landed experts and only upgrades idle ones);
a slot is reused only after the level-2 row that replaced it was applied; a failed read never posts its row, so the
expert simply stays at level 2."""
import torch,numpy as np
import p4rec as PR
from moe import entry

class RankExecutor:
    def __init__(s,rf,layers,nslot,n_host=64,qd=8,device=None,wait_for_slot=True):
        s.rf=rf;s.lay=rf.lay;s.rb=rf.rb;s.layers=layers
        dev=torch.device('cuda',torch.cuda.current_device() if device is None else device)
        s.slots=torch.empty(nslot,s.rb,dtype=torch.uint8,device=dev);s.free=list(range(nslot))[::-1];s.slot_of={}
        s.eng=rf.engine(n_host,qd,dev.index);s.tag=0;s.ops={}     # tag -> (L, E, kind, seq)
        s.wait_apply={}                                           # (L, E) -> (kind, seq) device writes done, not yet applied
        s.n_refused=0;s.n_failed=0;s.n_waited=0;s.lat=[];s.wait=wait_for_slot;s.pend=[]
    def _row(s,L,E,lv,slot=None):
        ex=s.layers[L][2][E]
        if lv==4:return PR.row(ex,s.lay,s.slots[slot].data_ptr(),entry)
        sg=ex.signs;ex.signs=ex.sc[2];r=entry(ex,2);ex.signs=sg;return r
    def apply(s,ups,downs,sched=None):
        for L,E in downs:
            M,MB,_=s.layers[L];s.tag+=1;q=MB.hseq[E]+1
            s.eng.post(s.tag,MB.stage[E].data_ptr(),s._row(L,E,2),MB.seq.data_ptr()+4*E,q);s.ops[s.tag]=(L,E,2,q)
        for L,E in ups:
            if not s.free and s.wait:s.pend.append((L,E));s.n_waited+=1;continue
            if not s.free:
                s.n_refused+=1
                if sched is not None:sched.failed(L,E)
                continue
            M,MB,_=s.layers[L];sl=s.free.pop();s.slot_of[L,E]=sl;s.tag+=1;q=MB.hseq[E]+1
            s.eng.upgrade(s.tag,s.rf.rec(L,E),s.slots[sl].data_ptr(),MB.stage[E].data_ptr(),s._row(L,E,4,sl),MB.seq.data_ptr()+4*E,q)
            s.ops[s.tag]=(L,E,4,q)
    def poll(s,sched=None):
        for tag,hit,trd,te2e in s.eng.poll():
            L,E,kind,q=s.ops.pop(tag)
            if trd<0:                                             # read failed: row never posted
                s.n_failed+=1;s.free.append(s.slot_of.pop((L,E)))
                if sched is not None:sched.failed(L,E,read_error=True)
                continue
            s.layers[L][1].hseq[E]=q;s.wait_apply[L,E]=(kind,q)
            if kind==4:s.lat.append(te2e)
        for (L,E),(kind,q) in list(s.wait_apply.items()):
            MB=s.layers[L][1]
            if int(MB.applied_host[E])!=q:continue
            del s.wait_apply[L,E]
            if kind==4:
                if sched is not None:sched.landed(L,E)
            else:
                s.free.append(s.slot_of.pop((L,E)))
                if sched is not None:sched.released(L,E)
        if s.pend and s.free:
            n=min(len(s.pend),len(s.free));go,s.pend=s.pend[:n],s.pend[n:];s.apply(go,[],sched)
    def busy(s):return bool(s.ops or s.wait_apply or s.pend)
    def close(s):s.eng.close()
