"""Level scheduler (work item B5): fixed set (always level 4, loaded at startup) + floating set chosen from decayed
routing counts. Pure policy: it sees per-step routing counts (the kernel's `hits` export, summed over the step's
tokens) and returns level ops; the executor carries them out. Every rank runs an identical copy on identical counts,
and the byte budget is per token (not wall time), so all ranks make the same decisions.

Defaults follow the lead (floating-set defaults 2026-09-28) and select_sweep.json:
  score = counts decayed with half-life 512 tokens (EMA 512), refresh every 64 tokens, 51 floating experts per layer;
  start (and any layer with no counts yet) from floating_default; upgrades over the byte budget are deferred (the
  expert stays at level 2) and served highest score first; no upgrades in a step that touches more than big_frac of
  a layer's experts; kv_pressure(n) drops the n lowest-score floating experts at once.
  Scheduler(layers, fixed, floating_default, rec_bytes, ...)
  step(counts[len(layers), NE], ntok) -> (ups [(L, E)], downs [(L, E)])   call once per model step
  landed(L, E) / released(L, E) / failed(L, E)   executor feedback: upgrade applied / downgrade applied (slot free) /
                                   upgrade refused or read failed (expert stays at level 2, retried later;
                                   after a read error not before retry_tokens)
  slots=N caps experts in flight + landed + draining at the executor's slot pool size."""
import numpy as np

class Scheduler:
    def __init__(s,layers,fixed,floating_default,rec_bytes,NE=256,n_float=51,half_life=512,refresh=64,
                 cap_GBps=6.0,tok_per_s=111.0,big_frac=0.5,burst_tokens=64,slots=None,retry_tokens=None):
        s.layers=list(layers);s.li={L:i for i,L in enumerate(s.layers)};s.NE=NE;s.nf=n_float;s.R=refresh
        s.a=0.5**(1/half_life);s.big=big_frac;s.rb=rec_bytes
        s.fixed=np.zeros((len(s.layers),NE),bool)
        for L in s.layers:s.fixed[s.li[L],list(fixed[L])]=True
        s.score=np.zeros((len(s.layers),NE))
        s.want=np.zeros_like(s.fixed)                      # floating target
        for L in s.layers:
            d=[e for e in floating_default[L] if not s.fixed[s.li[L],e]][:n_float];s.want[s.li[L],d]=True
        s.state=np.zeros((len(s.layers),NE),np.int8)       # floating: 0 level 2, 1 upgrade in flight, 2 level 4, 3 downgrade in flight
        s.per_tok=cap_GBps*1e9/tok_per_s;s.budget=0.0;s.cap=s.per_tok*burst_tokens
        s.slots=slots                                      # slot pool size (streamed experts per rank); None = unbounded
        s.retry=refresh if retry_tokens is None else retry_tokens;s.hold=np.zeros((len(s.layers),NE))   # failed read -> no retry before hold
        s.tok=0;s.next_refresh=refresh;s.stats=dict(ups=0,downs=0,deferred_steps=0,big_steps=0,bytes=0)
    def step(s,counts,ntok=1):
        c=np.asarray(counts,np.float64)
        s.score=s.score*s.a**ntok+c;s.tok+=ntok
        s.budget=min(s.cap,s.budget+s.per_tok*ntok)
        if s.tok>=s.next_refresh:
            s.next_refresh+=s.R
            has=s.score.sum(1)>0                           # layers with no counts keep floating_default
            sc=np.where(s.fixed,-np.inf,s.score);top=np.argsort(-sc,1,kind='stable')[:,:s.nf]
            w=np.zeros_like(s.want);np.put_along_axis(w,top,True,1);s.want[has]=w[has]
        downs=[(s.layers[i],int(e)) for i,e in zip(*np.nonzero((s.state==2)&~s.want))]
        for L,e in downs:s.state[s.li[L],e]=3
        ups=[]
        big=((c>0).sum(1)>s.big*s.NE).any()
        cand=(s.state==0)&s.want&(s.hold<=s.tok)
        if big:s.stats['big_steps']+=1
        elif cand.any():
            i,e=np.nonzero(cand);o=np.argsort(-s.score[i,e],kind='stable')
            n=int(s.budget//s.rb)
            if s.slots is not None:n=max(0,min(n,s.slots-int((s.state>0).sum())))   # slots held by in-flight, landed and draining
            take=o[:n]
            if len(o)>n:s.stats['deferred_steps']+=1
            for k in take:ups.append((s.layers[i[k]],int(e[k])));s.state[i[k],e[k]]=1
            s.budget-=len(take)*s.rb;s.stats['bytes']+=len(take)*s.rb
        s.stats['ups']+=len(ups);s.stats['downs']+=len(downs)
        return ups,downs
    def landed(s,L,e):
        i=s.li[L]
        if s.state[i,e]==1:s.state[i,e]=2
    def released(s,L,e):
        i=s.li[L]
        if s.state[i,e]==3:s.state[i,e]=0
    def failed(s,L,e,read_error=False):
        i=s.li[L]
        if s.state[i,e]==1:
            s.state[i,e]=0
            if read_error:s.hold[i,e]=s.tok+s.retry;s.stats['read_errors']=s.stats.get('read_errors',0)+1
    def kv_pressure(s,n):
        """drop the n lowest-score level-4 floating experts now (returns downs); they are not re-upgraded until the
        next refresh re-selects them."""
        i,e=np.nonzero(s.state==2)
        if not len(i):return []
        o=np.argsort(s.score[i,e],kind='stable')[:n];d=[(s.layers[i[k]],int(e[k])) for k in o]
        for L,x in d:s.state[s.li[L],x]=3;s.want[s.li[L],x]=False
        s.stats['downs']+=len(d);return d
    def level(s):
        """[len(layers), NE] level each expert is served at (as far as the scheduler knows)."""
        return np.where(s.fixed|(s.state==2),4,2)
