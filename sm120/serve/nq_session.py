"""Per-session floating-set restore for interleaved agent sessions (tb4 at c2: max_num_seqs=1, two agents alternating).

NQ_SESSION_RESTORE (default DEFAULT_ON below) | in-boot A/B: /dev/shm/nq_sr_ctl "on=0|1" (re-read at each request arrival;
the per-request log / NQ_SR_OUT jsonl are written in both modes, so off vs on is one boot).

Session key: blake2b of the prompt tokens up to (not incl.) the first <|assistant|> (154828), capped at NQ_SR_KEYCAP (8192)
tokens (no <|assistant|> in the cap: the first min(len, cap) tokens). In the GLM-5.3 chat template that is
[gMASK]<sop> + system + the first user message: terminus-2 puts its shared instructions (659 tokens common to all tasks)
AND the task description into that first user message, and every later turn of the task re-sends it verbatim, so the key
is constant within a task and different between tasks. A fixed prefix length can't do both (must be > 659 and <= 743,
the shortest first prompt seen). After a terminus-2 summarization the prefix changes: that request is an unknown session.

Per session (LRU of NQ_SR_LRU = 4): the floating set (bool [layers, NE] = record ids, not bytes), its upgrade order, the
predictor state (GBDT EMAs / block state / applied score matrix / the next_refresh result in flight; GBDTPredictorV2
salience EMAs too) and the scheduler's EMA score.

Rank 0 only (the scheduling rank); its ops go through the oplog like every other level op, so ranks 1-3 replay them.
Timeline of a request whose key differs from the previous request's (a 'switch'):
  arrival   (worker execute_model of the request's first step, i.e. before its LMCache load and prefill run): key,
            snapshot of the outgoing session -> LRU. Target set T = the saved set (known session) or the set at arrival
            (unknown session = the decode predictor's set, not the prefill-count set lookahead would leave behind).
  start     as soon as the remaining prefill is too short to hide the load: est. prefill time <= NQ_SR_MARGIN (2) x est.
            load time + NQ_SR_SLACK_S (0.5); est. load = diff x record bytes / NQ_SR_GBPS (2.5 GB/s per rank), prefill rate
            measured per request (NQ_SR_PF_TPS 1500 tok/s until measured). Issues the diff only: ups for T not resident,
            downs for resident floating not in T; ups ordered layer-first (the next forward pass needs the first layers
            first), then by the saved order score (highest predicted hits first). T is pinned from here to the handover:
            neither the host-loop scheduler nor lookahead down a pinned expert; lookahead keeps running with the leftover
            slots, its ups capped at NQ_SR_LA_BUDGET (8) per layer per chunk while restore ops are in flight (restore reads
            are queued first and win the slot pool; lookahead can't starve them).
  handover  first decode-like host-loop iteration after the request's prompt is fully scheduled: pin released, want := T,
            known session -> predictor state + EMA score restored. Decode never waits: records still in flight serve at
            level 2 and upgrade as they land.
Per request (switches and same-session requests alike) the log line / jsonl give arrival->start / ->landed / ->prefill end,
the landed-before-decode flag, ups/downs, and the served level-4 share of the first NQ_SR_WIN (64) decode tokens."""
import os,copy,json,time,hashlib,logging,collections
import numpy as np
try:
    from vllm.logger import init_logger;log=init_logger('vllm.nestquant')
except Exception:log=logging.getLogger('nestquant')

DEFAULT_ON='1'
ON=os.environ.get('NQ_SESSION_RESTORE',DEFAULT_ON)=='1'
CTL=os.environ.get('NQ_SR_CTL','/dev/shm/nq_sr_ctl')
ASSISTANT=154828
KEYCAP=int(os.environ.get('NQ_SR_KEYCAP','8192'));LRU=int(os.environ.get('NQ_SR_LRU','4'))
MARGIN=float(os.environ.get('NQ_SR_MARGIN','2'));SLACK=float(os.environ.get('NQ_SR_SLACK_S','0.5'))
GBPS=float(os.environ.get('NQ_SR_GBPS','2.5'));PF_TPS=float(os.environ.get('NQ_SR_PF_TPS','1500'))
LA_BUDGET=int(os.environ.get('NQ_SR_LA_BUDGET','8'));WIN=int(os.environ.get('NQ_SR_WIN','64'))
OUT=os.environ.get('NQ_SR_OUT','/dbg/sr_stats.jsonl')
DEC_MAX=32          # an iteration routing more than DEC_MAX tokens on some layer is prefill (residue), not decode
PSTATE=('E','Et','Ea','wt','wa','last','nblk','bc','bca','btok','bans','seg','S','h16','bst_state','Es','Ec','bs','s16')

def session_key(ids,cap=KEYCAP):
    """(hex key, prefix length) of a prompt token list"""
    try:k=ids.index(ASSISTANT,0,min(len(ids),cap))
    except ValueError:k=min(len(ids),cap)
    return hashlib.blake2b(np.asarray(ids[:k],np.int64).tobytes(),digest_size=8).hexdigest(),k

def _check_predictor(P):
    """Only the CPU predictors whose complete state schema is defined above.

    Joint/GPU predictors own additional history and CUDA-graph-bound buffers;
    restoring a partial generic attribute list silently corrupts their policy.
    They need their own snapshot protocol before session restore can be enabled.
    Check before draining refresh queues or changing scheduler residency.
    """
    if P is None:return
    kind=(type(P).__module__,type(P).__name__)
    if kind not in {('gbdt_predictor','GBDTPredictor'),
                    ('gbdt_predictor_v2','GBDTPredictorV2')}:
        raise RuntimeError(f'NestQuant session restore unsupported predictor {kind}; disable NQ_SESSION_RESTORE')

def _restore_value(current, saved):
    # Preserve array identities held by consumers, including nested EMA arrays.
    if isinstance(current,np.ndarray) and isinstance(saved,np.ndarray):
        if current.shape!=saved.shape or current.dtype!=saved.dtype:
            raise ValueError('NestQuant session state array schema changed')
        np.copyto(current,saved);return current
    if isinstance(current,list) and isinstance(saved,list):
        if len(current)!=len(saved):raise ValueError('NestQuant session state list schema changed')
        for i,v in enumerate(saved):current[i]=_restore_value(current[i],v)
        return current
    return copy.deepcopy(saved)

def p_snap(P):
    """copy of a GBDT predictor's decode state; drains (and keeps) a next_refresh result in flight"""
    if P is None:return None
    _check_predictor(P)
    pend=None
    if getattr(P,'_pending',False):pend=P._res.get();P._pending=False
    d={k:copy.deepcopy(getattr(P,k)) for k in PSTATE if hasattr(P,k)};d['_pend']=pend
    return d

def p_load(P,d):
    if P is None or d is None:return
    _check_predictor(P)
    if getattr(P,'_pending',False):P._res.get();P._pending=False
    for k,v in d.items():
        if k!='_pend':setattr(P,k,_restore_value(getattr(P,k,None),v))
    if d['_pend'] is not None:P._res.put(d['_pend']);P._pending=True

class SessionRestore:
    def __init__(s,rt=None):
        s.rt=rt;s.q=collections.deque();s.store=collections.OrderedDict();s.seen=set();s.cur=None;s.on=ON;s.ctl_m=None
        s.reqs={}           # hook side: req_id -> [prompt len, done]
        s.A=None            # the request being tracked (loop side)
        s.stats=collections.Counter();s.multi=False
        log.info('NestQuant session restore: %s (key = prompt up to the first <|assistant|>, cap %d; LRU %d; ctl %s)',
                 'on' if s.on else 'off (measure only)',KEYCAP,LRU,CTL)
    def _ctl(s):
        try:m=os.stat(CTL).st_mtime_ns
        except OSError:return
        if m==s.ctl_m:return
        s.ctl_m=m
        try:kv=dict(t.split('=',1) for t in open(CTL).read().split() if '=' in t)
        except (OSError,ValueError):return
        if 'on' in kv:s.on=kv['on']=='1';log.info('NestQuant session restore ctl: %s','on' if s.on else 'off')
        if kv.get('reset')=='1':s.store.clear();s.seen.clear();s.cur=None
    # ------------------------------------------------------------------ worker thread (execute_model, rank 0)
    def on_sched(s,so):
        """called at the start of every Worker.execute_model with the SchedulerOutput; cheap, never blocks"""
        t=time.time();ev=[];ns=so.num_scheduled_tokens;multi=len(ns)>1
        for rid in [r for r in s.reqs if r not in ns]:         # finished / aborted / preempted: no longer tracked
            del s.reqs[rid];ev.append(('finish',rid,t))
        new=set()
        for nr in so.scheduled_new_reqs:
            ids=nr.prompt_token_ids if nr.prompt_token_ids is not None else getattr(nr,'prefill_token_ids',None)
            if ids is None:continue
            k,kl=session_key(ids);n=ns.get(nr.req_id,0);new.add(nr.req_id)
            s.reqs[nr.req_id]=[len(ids),nr.num_computed_tokens+n]
            ev.append(('arrive',nr.req_id,k,kl,len(ids),nr.num_computed_tokens,n,t,multi))
        for rid,n in ns.items():
            r=s.reqs.get(rid)
            if r is None or rid in new:continue
            if r[1]>=r[0]:
                if len(r)==2:r.append(1);ev.append(('decode',rid,t))
            else:r[1]+=n;ev.append(('sched',rid,max(r[0]-r[1],0),t))
        if ev:
            s.q.extend(ev)
            if s.rt is not None:s.rt.wake.set()
    # ------------------------------------------------------------------ streaming thread (leader)
    def _snapshot(s,S):
        res=np.isin(S.state,(1,2))&~S.fixed
        osc=S.P.order_score(res) if S.P is not None and S.P.S is not None else None
        return dict(set=res.copy(),ord=np.array(S.score if osc is None else osc,np.float64),P=p_snap(S.P),score=S.score.copy())
    def service(s,S,X,oplog):
        A=s.A
        if A is not None and A['ups'] is not None and A['t_landed'] is None and not (S.state[A['ups']]==1).any():
            A['t_landed']=time.time();s._maybe_finish(S)
        while s.q:
            e=s.q.popleft();k=e[0]
            if k=='arrive':s._arrive(S,X,oplog,*e[1:])
            elif k=='sched':
                A=s.A
                if A is not None and A['rid']==e[1]:
                    A['pf'].append((e[3],e[2]));A['rem']=e[2]
                    (t0,r0),(t1,r1)=A['pf'][0],A['pf'][-1]
                    if t1>t0 and r0>r1:A['tps']=(r0-r1)/(t1-t0)
                    if A['pend_start']:
                        A['ndiff']=int((A['tgt']['set']&(S.state==0)&~S.fixed).sum())   # live diff: prefill lookahead moves the set
                        if s._due(A,X):s._start(S,X,oplog)
            elif k=='decode':
                A=s.A
                if A is not None and A['rid']==e[1] and A['t_dec'] is None:
                    A['t_dec']=e[2]
                    if A['pend_start']:s._start(S,X,oplog)      # restore not started yet: go now
            elif k=='finish':
                A=s.A
                if A is not None and A['rid']==e[1]:
                    A['done']=True
                    if not A['handed']:s._finish(S,'finished before decode')
                    else:s._maybe_finish(S)
    def _arrive(s,S,X,oplog,rid,key,kl,plen,ncomp,nsched,t,multi):
        s._ctl()
        if s.on and not multi:_check_predictor(S.P)
        if s.A is not None:s._finish(S,'superseded')
        known=key in s.store;seen=key in s.seen;switch=key!=s.cur;on=s.on and not multi;s.seen.add(key)
        if multi and s.on and not s.multi:
            s.multi=True;log.warning('NestQuant session restore: several running requests, restore skipped for them (max_num_seqs=1 is the supported case)')
        A=dict(rid=rid,key=key,klen=kl,plen=plen,ncomp=ncomp,t_arr=t,known=known,seen=seen,switch=switch,on=on,rem=plen-ncomp-nsched,
               rem0=plen-ncomp,pf=[(t,plen-ncomp-nsched)],tps=None,done=False,pend_start=False,tgt=None,ups=None,t_start=None,t_landed=None,
               t_dec=None,t_hand=None,n_up=0,n_down=0,w_tok=0,w4=0,w_all=0,handed=False,la_ups0=S.stats.get('la_ups',0))
        s.A=A;s.stats['requests']+=1
        if switch:s.stats['switches']+=1;s.stats['seen' if seen else 'new']+=1
        if switch and on:
            out=s._snapshot(S)
            if s.cur is not None:
                s.store[s.cur]=out;s.store.move_to_end(s.cur)
                while len(s.store)>LRU:s.store.popitem(last=False)
            A['tgt']=s.store[key] if known else out
            A['own']=out              # the outgoing session's state (the unknown-session fallback restores this one)
            T=A['tgt']['set'];A['ndiff']=int((T&(S.state==0)&~S.fixed).sum());A['pend_start']=True
            if s._due(A,X):s._start(S,X,oplog)
        s.cur=key
    def _due(s,A,X):
        tl=A['ndiff']*X.rb/(GBPS*1e9);tp=A['rem']/(A['tps'] or PF_TPS)
        return tp<=MARGIN*tl+SLACK
    def _start(s,S,X,oplog):
        A=s.A;A['pend_start']=False;T=A['tgt']['set'];o=A['tgt']['ord'];fx=S.fixed
        um=T&(S.state==0)&~fx&(S.hold<=S.tok);dm=(S.state==2)&~T&~fx
        i,e=np.nonzero(um);r=np.lexsort((-o[i,e],i))                   # layer first, then highest order score
        ups=[(S.layers[i[k]],int(e[k])) for k in r]
        j,f=np.nonzero(dm);downs=[(S.layers[a],int(b)) for a,b in zip(j,f)]
        S.state[um]=1;S.state[dm]=3;S.pin=T.copy()
        S.stats['ups']+=len(ups);S.stats['downs']+=len(downs);S.stats['bytes']+=len(ups)*S.rb
        S.stats['sr_ups']=S.stats.get('sr_ups',0)+len(ups)
        if ups or downs:
            X.apply(ups,downs,S)
            if oplog is not None:oplog.put(ups,downs)
        A['ups']=um;A['n_up']=len(ups);A['n_down']=len(downs);A['t_start']=time.time()
        if not ups:A['t_landed']=A['t_start']
    def restoring(s):
        """restore reads in flight (lookahead caps its per-chunk ups while True)"""
        A=s.A;return A is not None and A['ups'] is not None and A['t_landed'] is None
    def on_counts(s,S,c,ntok,lv):
        """host loop, rank 0, before S.step on counts c [layers, NE]; lv = mask served at level 4 this step.
        Returns ntok to pass to S.step (prefill residue after the prompt is fully scheduled is stepped as prefill)."""
        A=s.A
        if A is None:return ntok
        big=int(c.sum(1).max())>TOPK*DEC_MAX
        if not A['handed']:
            if A.get('t_dec') is None or big or ntok<=0:return max(ntok,int(c.sum(1).max())//TOPK) if big else ntok
            s._handover(S);lv=S.fixed|(S.state==2)
        elif big:return max(ntok,int(c.sum(1).max())//TOPK)
        if A['w_tok']<WIN:
            A['w_tok']+=ntok;A['w4']+=int(c[lv].sum());A['w_all']+=int(c.sum())
        s._maybe_finish(S)
        return ntok
    def _maybe_finish(s,S):
        A=s.A        # log once the window is full and the restore reads are done (or the request ended)
        if A['handed'] and (A['w_tok']>=WIN or A['done']) and (A['t_landed'] is not None or A['ups'] is None):s._finish(S,'ok')
    def _handover(s,S):
        A=s.A;A['handed']=True;A['t_hand']=time.time()
        if A['tgt'] is not None:
            if A['pend_start']:A['pend_start']=False
            S.pin=None;S.want=A['tgt']['set']&~S.fixed
            if A['known']:p_load(S.P,A['tgt']['P']);S.score=A['tgt']['score'].copy()
            else:p_load(S.P,A['own']['P'])            # state is unchanged since arrival: put back the drained refresh result
    def _finish(s,S,why):
        A=s.A
        if A is None:return
        if not A['handed'] and A['tgt'] is not None:        # never reached decode (aborted / superseded): drop the pin, keep T wanted
            S.pin=None
            if not A['known']:p_load(S.P,A['own']['P'])
            else:p_load(S.P,A['tgt']['P'])
        if A['ups'] is not None and A['t_landed'] is None and not (S.state[A['ups']]==1).any():A['t_landed']=time.time()
        s.A=None
        f=lambda x:None if x is None else round((x-A['t_arr'])*1e3,1)
        land,dec=A['t_landed'],A['t_hand']
        ov=None
        if land is not None and dec is not None:ov=round(min(1.,(min(land,dec)-A['t_arr'])/max(land-A['t_arr'],1e-9)),3)
        r=dict(t=round(A['t_arr'],3),key=A['key'],klen=A['klen'],plen=A['plen'],cached=A['ncomp'],new=A['rem0'],switch=A['switch'],
               known=A['seen'],restored=A['known'] and A['tgt'] is not None,on=A['on'],ups=A['n_up'],downs=A['n_down'],ms_start=f(A['t_start']),ms_landed=f(land),
               ms_last_prefill_sched=f(A['t_dec']),ms_decode_start=f(A['t_hand']),landed_before_decode=None if land is None or dec is None else land<=dec,
               overlap=ov,pf_tps=None if A['tps'] is None else round(A['tps']),la_ups=S.stats.get('la_ups',0)-A['la_ups0'],
               win_tok=A['w_tok'],win_share4=round(A['w4']/A['w_all'],4) if A['w_all'] else None,end=why)
        log.info('NestQuant session: %s',json.dumps(r))
        try:
            with open(OUT,'a') as fo:fo.write(json.dumps(r)+'\n')
        except OSError:pass

TOPK=8
