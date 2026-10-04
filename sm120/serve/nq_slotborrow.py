"""NQ_LMPF slot borrow (phase 1): during an LMPF prefill the ring (2 x C records) and, with NQ_LMPF_BORROW=all, the
window state live in one contiguous span of the executor's slot pool (idle decode slots) instead of their own memory.
Protocol (every TP rank, same step, after the TP vote; the caller holds the executor fence = rt.ncap and the loop hold
= rt.lm_hold, so nothing else touches X / S / F):
  drain  every MoE mailbox applied (direct), stream synced, X.poll(issue=False): landings / releases finalised;
         the executor must be quiet (no op in flight, nothing waiting for an apply, no prefill-borrow pool)
  r1     all-reduce MAX [cant, V0]: V0 = the leader's lowest-score floating level-4 residents, as many as it lacks
         free slots for (executor and scheduler view) + MARGIN
  r2     all-reduce MAX [extras]: residents a rank must evict beyond V0 to have n free slots (normally none)
  evict  U = V0 | extras on every rank: waiting ups cancelled, residents' rows -> level 2 (staged, applied, synced),
         slots freed. Leader: S.evict(U) (state 0, tap doom / todo purged) + an oplog downs record of U, which is a
         no-op on the followers (released + nl here) and neutralises their older, not yet replayed ups of U
  span   per rank the n-slot window with the fewest occupants; they are relocated (D2D copy + level-4 row switch,
         applied + synced) to free slots outside it; X.lend(a, n) (never free / released until return);
         leader S.slots -= n
  return (give_back) after the caller drained the ring and synced the stream: X.unlend, leader S.slots += n,
         want = D0 (pre-borrow residents; base scheduler only); refill = normal scheduling + oplog, no extra traffic"""
import os,time,numpy as np,torch
MARGIN=int(os.environ.get('NQ_LMPF_BORROW_MARGIN','8'))

def leader_victims(S,X,n,margin=MARGIN):
    """[nL, NE] bool: the leader's lowest-score floating level-4 residents, as many as it is short of n free slots
    (executor free list and scheduler budget, whichever is shorter) + margin; none if it already has n free"""
    st=S.state;m=np.zeros(st.shape,bool)
    need=n-len(X.free)
    if S.slots is not None:need=max(need,n-(int(S.slots)-int((st>0).sum())))
    if need<=0:return m
    i,e=np.nonzero((st==2)&~S.fixed)
    o=np.argsort(S.score[i,e],kind='stable')[:need+margin];m[i[o],e[o]]=True
    return m

def rank_extras(X,n,V0,li):
    """residents (lowest slot first) this rank evicts beyond V0 so that free + V0-held >= n"""
    held=[(sl,k) for k,sl in X.slot_of.items() if sl<X.nslot]
    have=len(X.free)+sum(1 for sl,k in held if V0[li[k[0]],k[1]])
    if have>=n:return []
    return [k for sl,k in sorted(held) if not V0[li[k[0]],k[1]]][:n-have]

def pick_span(X,n):
    """start of the n-slot window of the pool with the fewest occupants (lent slots excluded)"""
    occ=np.zeros(X.nslot,np.int64)
    for sl in X.slot_of.values():
        if sl<X.nslot:occ[sl]=1
    for sl in X.lent:occ[sl]=1<<40
    c=np.concatenate([[0],np.cumsum(occ)]);w=c[n:]-c[:X.nslot-n+1]
    return int(np.argmin(w))

def _drain(rt,sched):
    X=rt.X
    for L in X.layers:X.layers[L][1].apply()
    torch.cuda.synchronize(X.slots.device)
    X.poll(sched,issue=False)

def borrow(rt,n,ar,cant=False,margin=MARGIN):
    """collective (see module doc). ar(int32 array) -> element-wise MAX over the TP ranks. -> info dict or None
    (None on every rank when any rank cannot borrow)"""
    X=rt.X;S=rt.S;lead=rt.F is None;sched=S if lead else rt.F;t0=time.perf_counter()
    nL,NE=S.state.shape;li=S.li
    _drain(rt,sched)
    ok=(not cant) and X.quiet() and not X.lent and 0<n<=X.nslot
    v=np.zeros(1+nL*NE,np.int32);v[0]=0 if ok else 1
    if ok and lead:v[1:]=leader_victims(S,X,n,margin).reshape(-1)
    v=ar(v)
    if v[0]:return None
    V0=v[1:].reshape(nL,NE).astype(bool)
    w=np.zeros(nL*NE,np.int32)
    for L,E in rank_extras(X,n,V0,li):w[li[L]*NE+E]=1
    w=ar(w)
    Um=V0|w.reshape(nL,NE).astype(bool);lay=S.layers
    U=[(lay[i],int(e)) for i,e in zip(*np.nonzero(Um))];Us=set(U)
    D0=(np.isin(S.state,(1,2))&~S.fixed) if lead else None
    for k in [k for k in X.pend if k in Us]:X.cancel_up(*k,sched)
    ev=X.evict_now(U)
    if lead:
        S.evict(U)
        if rt.log is not None and U:rt.log.put([],U)
    else:
        for k in ev:rt.F.released(*k)
        nl=getattr(rt.F,'nl',None)
        if nl is not None:nl.update(U)
    a=pick_span(X,n)
    mv=sorted((sl,k) for k,sl in X.slot_of.items() if a<=sl<a+n)
    dst=[f for f in reversed(X.free) if not a<=f<a+n][:len(mv)]
    assert len(dst)==len(mv),('slot borrow: no room to relocate',len(mv),len(dst))
    X.relocate([(k,sl,d) for (sl,k),d in zip(mv,dst)])
    X.lend(a,n)
    if lead and S.slots is not None:S.slots-=n
    return dict(a=a,n=n,U=len(U),ev=len(ev),moved=len(mv),ms=(time.perf_counter()-t0)*1e3,D0=D0)

def give_back(rt,info):
    """return the span (the caller drained every reader of it). -> slots returned"""
    X=rt.X;n=X.unlend()
    if rt.F is None:
        S=rt.S
        if S.slots is not None:S.slots+=n
        D0=info.get('D0') if info else None
        if D0 is not None and getattr(S,'pin',None) is None and not hasattr(S,'doom'):S.want=D0.copy()
    return n

class ColPlan:
    """online first-fit byte columns per token: items arrive in non-decreasing start order (start, end, key, bytes/token);
    an item occupies the boundaries start .. end-1; overlapping items get disjoint columns. stride = the per-token width
    available (None = unbounded). add() -> (col, width) or None when it does not fit in the stride"""
    def __init__(s,stride=None,align=16):
        s.stride=stride;s.align=align;s.act=[];s.cols={};s.tot=0
    def add(s,st,en,key,b):
        wd=-(-int(b)//s.align)*s.align;s.act=[x for x in s.act if x[1]>st];c=0
        for a0,a1 in sorted((x[2],x[2]+x[3]) for x in s.act):
            if c+wd<=a0:break
            c=max(c,a1)
        if s.stride is not None and c+wd>s.stride:return None
        s.act.append((st,en,c,wd));s.cols[key]=(c,wd);s.tot=max(s.tot,c+wd)
        return c,wd

def plan_cols(items,align=16):
    """static arena columns for the compiled exec: items [(start, end, key, bytes/token)] = value produced in row start,
    last used in row end (> start): it occupies the boundaries start .. end-1 (after row b). Values whose boundary
    ranges overlap get disjoint byte columns. -> ({key: (col, width)}, per-token bytes); cols / widths % align == 0"""
    P=ColPlan(None,align)
    for st,en,key,b in sorted(items,key=lambda x:(x[0],-x[3],str(x[2]))):P.add(st,en,key,b)
    return P.cols,P.tot
