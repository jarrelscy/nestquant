"""Opt-in diagnostic observer; no scheduling decisions or CUDA synchronization.
Times are host-observed bounds, not device timestamps. Bounded samples only.
"""
import collections, functools, json, os, time
from pathlib import Path

class Observer:
    def __init__(self):
        self.issue={};self.complete={};self.count=collections.Counter()
        self.samples={k:collections.deque(maxlen=4096) for k in ('issue_to_complete_ms','complete_to_ack_ms','issue_to_ack_ms')}
        self.last=0.
    def summary(self,ex,pool,now):
        def stats(xs):
            v=sorted(xs)
            return {'n':len(v),'p50':v[len(v)//2] if v else None,'p99':v[min(len(v)-1,int(len(v)*.99))] if v else None,'max':v[-1] if v else None}
        return {'time':time.time(),'clock':'host monotonic observations; completion/ack may predate observation',
                'counts':dict(self.count),'latency_ms':{k:stats(v) for k,v in self.samples.items()},
                'pending_issue':len(self.issue),'pending_completed':len(self.complete),
                'oldest_issue_ms':max((now-t for t in self.issue.values()),default=0)*1000,
                'oldest_completed_ms':max((now-t for t in self.complete.values()),default=0)*1000,
                'wait_apply_up':sum(kind==4 for kind,q in ex.wait_apply.values()),
                'wait_apply_down':sum(kind!=4 for kind,q in ex.wait_apply.values()),
                'free_slots':len(ex.free), 'worker_timing':{k:v for k,v in ex.eng.stats().items() if k.startswith('diag_')}}

def patch_executor(module):
    cls=module.RankExecutor
    original_init,original_apply,original_poll=cls.__init__,cls.apply,cls.poll
    @functools.wraps(original_init)
    def init(self,*a,**k):
        original_init(self,*a,**k);self._publication_observer=Observer()
    @functools.wraps(original_apply)
    def apply(self,*a,**k):
        before=set(self.ops)|set(self.wait_apply);result=original_apply(self,*a,**k)
        ob=self._publication_observer;now=time.monotonic()
        # ops indexed by unique engine tag, wait_apply by (layer,expert).
        for tag,op in self.ops.items():
            if tag not in before:
                L,E,kind,q=op;key=(L,E,kind,q)
                if key not in ob.issue:ob.issue[key]=now;ob.count['issued_up' if kind==4 else 'issued_down']+=1
        for (L,E),(kind,q) in self.wait_apply.items():
            key=(L,E,kind,q)
            if key not in ob.issue:ob.issue[key]=now;ob.count['issued_up' if kind==4 else 'issued_down']+=1
        return result
    @functools.wraps(original_poll)
    def poll(self,sched=None,*a,**k):
        ob=self._publication_observer;before=dict(self.ops);wait_before=dict(self.wait_apply)
        result=original_poll(self,sched,*a,**k);now=time.monotonic()
        for tag,(L,E,kind,q) in before.items():
            if tag in self.ops:continue
            key=(L,E,kind,q)
            # A failed/cancelled operation has no matching mailbox sequence.
            if self.layers[L][1].hseq[E]!=q:
                ob.issue.pop(key,None);ob.complete.pop(key,None);ob.count['cancelled_or_failed']+=1;continue
            ob.complete.setdefault(key,now);ob.count['engine_completion_observed']+=1
            if key in ob.issue:ob.samples['issue_to_complete_ms'].append((now-ob.issue[key])*1000)
        keys={(L,E,kind,q) for (L,E),(kind,q) in wait_before.items()}|set(ob.complete)
        for key in keys:
            L,E,kind,q=key
            if self.ah[L][E]!=q:continue
            if key in ob.complete:ob.samples['complete_to_ack_ms'].append((now-ob.complete.pop(key))*1000)
            if key in ob.issue:ob.samples['issue_to_ack_ms'].append((now-ob.issue.pop(key))*1000)
            ob.count['ack_up' if kind==4 else 'ack_down']+=1
            if kind==4 and sched is not None and not bool(sched.wanted[sched.li[L],E]):ob.count['ack_up_no_longer_wanted']+=1
        if now-ob.last>=2:
            ob.last=now;path=os.environ.get('NQ_PUBLICATION_OBSERVER')
            if path:
                p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
                with p.open('a') as f:f.write(json.dumps(ob.summary(self,sched,now))+'\n')
        return result
    cls.__init__,cls.apply,cls.poll=init,apply,poll
