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
  slots=N caps experts in flight + landed + draining at the executor's slot pool size.

Predictor (floating-set target): predictor='ema' (the rule above) or 'gbdt' (gbdt_predictor.GBDTPredictor, LightGBM
p64 model, 16-token refresh, next_refresh mode; see sm120/eval/PREDICTOR.md). Default from env NQ_PREDICTOR (else
DEFAULT_PREDICTOR). With gbdt the EMA score is still kept (kv_pressure, and candidate order until the first GBDT score
matrix exists); the target set and the upgrade order come from the predictor, and the budget / big-step guard / slots /
executor feedback are unchanged. step(..., token_ids=None, new_request=False) feeds the think/answer segment state
(None = segment stays 'think'). A predictor object with the GBDTPredictor interface can be passed instead of a name.
GBDT env: NQ_GBDT_MODE=next_refresh|sync, NQ_GBDT_SCALE=none|mps (mps = predicted hits x mps128, needs step(..., sal=)),
NQ_GBDT_BAND=ema256|all (all = score every non-fixed expert from EMA256 rank 20 on, not 20..120), NQ_GBDT_MODEL (a 9-feature
v2 model also selects GBDTPredictorV2), NQ_GBDT_THREADS."""
import os
import numpy as np

DEFAULT_PREDICTOR='gbdt'
def _nfeat(path):
    with open(path) as f:
        for ln in f:
            if ln.startswith('feature_names='):return len(ln.split('=',1)[1].split())
    return None
def make_predictor(name,layers,fixed,n_float=51,NE=256,**kw):
    if name in (None,'ema'):return None
    if name=='gbdt':
        kw.setdefault('num_threads',int(os.environ.get('NQ_GBDT_THREADS','4')))
        kw.setdefault('mode',os.environ.get('NQ_GBDT_MODE','next_refresh'))
        scale=kw.pop('scale',os.environ.get('NQ_GBDT_SCALE','none'));scale=None if scale in (None,'','none') else scale
        band=kw.pop('band',os.environ.get('NQ_GBDT_BAND','ema256'))   # ema256 = score EMA256 ranks 20..120; all = every non-fixed rank >= 20
        assert band in ('ema256','all'),band
        if band=='all':kw.setdefault('rhi',NE)
        mp=kw.pop('model_path',None) or os.environ.get('NQ_GBDT_MODEL') or None
        from gbdt_predictor import GBDTPredictor
        if scale is None and (mp is None or _nfeat(mp)==5):
            return GBDTPredictor(layers,fixed,model_path=mp,n_float=n_float,**kw)
        from gbdt_predictor_v2 import GBDTPredictorV2     # salience inputs: step(..., sal=[NL,NE])
        return GBDTPredictorV2(layers,fixed,model_path=mp,scale=scale,n_float=n_float,**kw)
    raise ValueError(f'unknown predictor {name!r}')

class Scheduler:
    def __init__(s,layers,fixed,floating_default,rec_bytes,NE=256,n_float=51,half_life=512,refresh=64,
                 cap_GBps=6.0,tok_per_s=111.0,big_frac=0.5,burst_tokens=64,slots=None,retry_tokens=None,
                 predictor=None,predictor_kw=None,hostloop=None):
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
        if predictor is None:predictor=os.environ.get('NQ_PREDICTOR') or DEFAULT_PREDICTOR
        s.predictor_name=predictor if isinstance(predictor,str) else type(predictor).__name__
        s.P=make_predictor(predictor,s.layers,fixed,n_float,NE=NE,**(predictor_kw or {})) if isinstance(predictor,str) else predictor
        s.wants_sal=s.P is not None and hasattr(s.P,'bs')              # GBDTPredictorV2 accumulates salience
        s.pin=None          # optional [len(layers), NE] bool: floating experts kept wanted (never downed) while set (session restore)
        # nq-io upgrade 5: NQ_HOSTLOOP=cpp runs the mechanical part of step() in C++ (nqhost.SchedCore, bit-exact:
        # tests/test_hostloop_parity.py); hostloop= overrides the env (tests)
        hl=hostloop if hostloop is not None else os.environ.get('NQ_HOSTLOOP','py')
        s.core=None
        if hl=='cpp':
            import hostcore;s.core=hostcore.mod().SchedCore(s.layers,NE)
    def step(s,counts,ntok=1,token_ids=None,new_request=False,sal=None):
        """sal: optional [len(layers),NE] per-step salience (sum over the step's routed slots of w^2*|x|^2), forwarded
        to predictors that take it (GBDTPredictorV2); ignored otherwise."""
        if s.core is not None:return s._step_cpp(counts,ntok,token_ids,new_request,sal)
        c=np.asarray(counts,np.float64)
        s.score=s.score*s.a**ntok+c;s.tok+=ntok
        s.budget=min(s.cap,s.budget+s.per_tok*ntok)
        osc=None
        if s.P is not None:
            if (s.P.step(c,ntok,token_ids,new_request,sal=sal) if s.wants_sal else s.P.step(c,ntok,token_ids,new_request)):
                res=np.isin(s.state,(1,2));w=s.P.target(res)
                if w is not None:s.want=w&~s.fixed
        elif s.tok>=s.next_refresh:
            s.next_refresh+=s.R
            has=s.score.sum(1)>0                           # layers with no counts keep floating_default
            sc=np.where(s.fixed,-np.inf,s.score);top=np.argsort(-sc,1,kind='stable')[:,:s.nf]
            w=np.zeros_like(s.want);np.put_along_axis(w,top,True,1);s.want[has]=w[has]
        if s.pin is not None:s.want|=s.pin&~s.fixed
        downs=[(s.layers[i],int(e)) for i,e in zip(*np.nonzero((s.state==2)&~s.want))]
        for L,e in downs:s.state[s.li[L],e]=3
        ups=[]
        big=((c>0).sum(1)>s.big*s.NE).any()
        cand=(s.state==0)&s.want&(s.hold<=s.tok)
        if big:s.stats['big_steps']+=1
        elif cand.any():
            if s.P is not None:osc=s.P.order_score(np.isin(s.state,(1,2)))
            i,e=np.nonzero(cand);o=np.argsort(-(s.score if osc is None else osc)[i,e],kind='stable')
            n=int(s.budget//s.rb)
            if s.slots is not None:n=max(0,min(n,s.slots-int((s.state>0).sum())))   # slots held by in-flight, landed and draining
            take=o[:n]
            if len(o)>n:s.stats['deferred_steps']+=1
            for k in take:ups.append((s.layers[i[k]],int(e[k])));s.state[i[k],e[k]]=1
            s.budget-=len(take)*s.rb;s.stats['bytes']+=len(take)*s.rb
        s.stats['ups']+=len(ups);s.stats['downs']+=len(downs)
        return ups,downs
    def _step_cpp(s,counts,ntok,token_ids,new_request,sal):
        """step() with the array work in nqhost.SchedCore (same ops, same order, same score / state bits)"""
        K=s.core;c=np.ascontiguousarray(counts,np.float64)
        for n,dt in (('score',np.float64),('want',bool),('state',np.int8),('hold',np.float64)):   # other modules may rebind these
            a=getattr(s,n)
            if a.dtype!=dt or not a.flags.c_contiguous or not a.flags.writeable:setattr(s,n,np.ascontiguousarray(a,dt).copy())
        K.decay(s.score,c,s.a**ntok);s.tok+=ntok
        s.budget=min(s.cap,s.budget+s.per_tok*ntok)
        osc=None
        if s.P is not None:
            if (s.P.step(c,ntok,token_ids,new_request,sal=sal) if s.wants_sal else s.P.step(c,ntok,token_ids,new_request)):
                w=s.P.target(K.resident(s.state))
                if w is not None:s.want=np.ascontiguousarray(w&~s.fixed)
        elif s.tok>=s.next_refresh:
            s.next_refresh+=s.R;K.ema_refresh(s.score,s.fixed,s.want,s.nf)
        if s.pin is not None:s.want|=s.pin&~s.fixed
        downs=K.downs(s.state,s.want)
        ups=[]
        big,anyc=K.select(c,s.state,s.want,s.hold,float(s.tok),s.big*s.NE)
        if big:s.stats['big_steps']+=1
        elif anyc:
            if s.P is not None:osc=s.P.order_score(K.resident(s.state))
            n=int(s.budget//s.rb)
            if s.slots is not None:n=max(0,min(n,s.slots-K.count_busy(s.state)))
            key=s.score if osc is None else np.ascontiguousarray(osc)
            r=K.take(key,s.state,n)
            if r is None:                                  # key dtype other than f64/f32: numpy ordering
                cand=(s.state==0)&s.want&(s.hold<=s.tok);i,e=np.nonzero(cand);o=np.argsort(-key[i,e],kind='stable')
                take=o[:n];more=len(o)>n
                for k in take:ups.append((s.layers[i[k]],int(e[k])));s.state[i[k],e[k]]=1
            else:ups,more=r
            if more:s.stats['deferred_steps']+=1
            s.budget-=len(ups)*s.rb;s.stats['bytes']+=len(ups)*s.rb
        s.stats['ups']+=len(ups);s.stats['downs']+=len(downs)
        return ups,downs
    def close(s):
        if s.P is not None:s.P.close()
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
    def evict(s,keys):
        """nq-lmpf slot borrow: keys were forced to level 2 by the executor (rows applied, slots released) -> state 0;
        tap: lazy evictions / queued promotions touching them are dropped"""
        U=set()
        for L,e in keys:
            i=s.li[L];s.state[i,e]=0;U.add((i,e))
        dm=getattr(s,'doom',None)
        if dm:
            for (i,e),(j,v) in list(dm.items()):
                if (i,e) in U or (j,v) in U:del dm[i,e];s.doomed[j,v]=False
        td=getattr(s,'todo',None)
        if td:
            k=[x for x in td if tuple(x) not in U]
            if len(k)!=len(td):td.clear();td.extend(k)
        s.stats['lend_evict']=s.stats.get('lend_evict',0)+len(U)
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
