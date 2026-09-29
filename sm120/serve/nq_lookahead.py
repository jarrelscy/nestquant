"""Prefill expert-level adaptation for the NQ serve (rank 0 decides, ranks 1-3 replay its ops from the oplog).

NQ_PREFILL_ADAPT = 0 (off) | lookahead | chunk
  lookahead: at MoE layer L of a prefill chunk (T >= NQ_PF_MIN tokens) apply layer L+d's own router (its GateLinear +
             router.select_experts, i.e. exactly vLLM's sigmoid + e_score_correction_bias grouped top-8) to x_L, score the
             experts of L+d, and upgrade its top non-resident ones right away (reads land before L+d runs, or the
             expert runs at level 2 this chunk: prefill never waits).
  chunk:     the same, but from layer L's actual routing of this chunk, for layer L in the next chunk.
NQ_LA_D (2)          lookahead distance d
NQ_LA_BUDGET (45)    upgrades per layer per chunk (bounded by free slots of the pool)
NQ_PF_RANK (gate)    ranking score over the tokens routed to e in the chunk: gate = sum_t w_te, count = routes,
                     salience = delta_e * sum_t w_te^2 ||x_t||^2 (delta from NQ_DELTA_TABLE, nq-delta-v1; without the
                     table delta = 1). Ranked within each layer.
NQ_LA_MEASURE (0)    1: router lookahead accuracy (d = 0, 1, 2 vs previous chunk / static set / oracle, k = 26/45/77, shares
                     of routes / gate weight / sum w^2||x||^2 / delta-benefit captured), json to NQ_LA_OUT; adds a host sync
                     per layer (measurement only).
Consistency: only the scheduling rank (rank 0 with NQ_LEADER) scores and plans; its ups/downs go through the oplog, so
all ranks carry out the same level ops (ranks 1-3 never run the router). Host transfer: the [8, 256] per-layer stats are
copied non-blocking into pinned memory + a CUDA event; the streaming thread picks them up (no extra sync in forward).
The byte budget (NQ_CAP_GBPS) and the scheduler's big-step guard do not apply to these ops (slot pool only).
Per chunk the streaming thread logs: issued / landed in time (served at level 4 by the layer the plan was for) /
deferred (wanted but no slot or still draining), the route / gate / salience share of the chosen set and of the served
level-4 set, and the router cost (CUDA events)."""
import os,re,json,time,collections,logging
import numpy as np,torch
try:
    from vllm.logger import init_logger;log=init_logger('vllm.nestquant')
except Exception:log=logging.getLogger('nestquant')
NE=256
MODE=os.environ.get('NQ_PREFILL_ADAPT','0')
if MODE in ('','0','off'):MODE=None
assert MODE in (None,'lookahead','chunk'),MODE
D=int(os.environ.get('NQ_LA_D','2'));BUDGET=int(os.environ.get('NQ_LA_BUDGET','45'))
RANK=os.environ.get('NQ_PF_RANK','gate');assert RANK in ('gate','count','salience'),RANK
MEAS=os.environ.get('NQ_LA_MEASURE','0')=='1'
DT=os.environ.get('NQ_DELTA_TABLE','');OUT=os.environ.get('NQ_LA_OUT','/dbg/la_stats.json')
KS=(26,45,77);MEAS_N=('count','gate','sal','dsal')   # sal = sum w^2 ||x||^2, dsal = delta * sal (the lead's 'sal')
_run={}             # layer -> vLLM MoERunner (gate + router)
GAMMA=os.environ.get('NQ_LA_GAMMA','1')!='0'   # rescale x_L to layer L+d's post_attention_layernorm (x * g_{L+d} / g_L)

def load_gamma(layers,dev):
    """post_attention_layernorm weights from the checkpoint (MODEL_DIR): RMSNorm is scale invariant, so
    norm_{L+d}(h) = x_L * g_{L+d} / g_L exactly for the residual h that layer L's MoE input x_L was normalized from"""
    md=os.environ.get('MODEL_DIR')
    if not md or not os.path.exists(md+'/model.safetensors.index.json'):return {}
    from safetensors import safe_open
    wm=json.load(open(md+'/model.safetensors.index.json'))['weight_map'];by=collections.defaultdict(list);g={}
    for L in layers:
        k=f'model.layers.{L}.post_attention_layernorm.weight'
        if k in wm:by[wm[k]].append((L,k))
    for f,ks in by.items():
        with safe_open(md+'/'+f,'pt') as h:
            for L,k in ks:g[L]=h.get_tensor(k).float().to(dev)
    return g

def install():
    """record every MoERunner by layer index (called at nq_vllm import, before the first NQ layer's runner exists)"""
    try:from vllm.model_executor.layers.fused_moe.runner import moe_runner as MR
    except Exception as e:log.warning('NestQuant lookahead: no MoERunner (%s)',e);return
    C=MR.MoERunner
    if getattr(C,'_nq_la',False):return
    i0=C.__init__
    def init(s,*a,**k):
        i0(s,*a,**k)
        m=re.search(r'layers\.(\d+)\.',str(k.get('layer_name') or getattr(s,'layer_name','')))
        if m:_run[int(m.group(1))]=s
    C.__init__=init;C._nq_la=True

def route(L,x):
    """layer L's router on x (bf16 [T, H]) -> (ids [T, 8] long, w [T, 8] f32) or None"""
    r=_run.get(L)
    if r is None or getattr(r,'gate',None) is None:return None
    lg,_=r.gate(x);R=r.router;cf=getattr(R,'capture_fn',None)
    if cf is not None:R.capture_fn=None            # never record lookahead routes as the layer's routed experts
    try:w,ids=R.select_experts(hidden_states=x,router_logits=lg)
    finally:
        if cf is not None:R.capture_fn=cf
    return ids.long(),w.float()

def stats(ids,w,x2,delta=None):
    """per-expert [4, NE] f32: routes, sum w, sum w^2 ||x||^2, delta * sum w^2 ||x||^2"""
    S=torch.zeros(4,NE,dtype=torch.float32,device=ids.device);f=ids.flatten()
    S[0].scatter_add_(0,f,torch.ones(f.numel(),dtype=torch.float32,device=f.device))
    S[1].scatter_add_(0,f,w.flatten());S[2].scatter_add_(0,f,(w*w*x2[:,None]).flatten())
    S[3]=S[2]*delta if delta is not None else S[2]
    return S

class LA:
    def __init__(s,rt):
        s.rt=rt;s.on=MODE is not None;s.meas=MEAS;s.layers=sorted(rt.lay);s.dev=rt.dev
        s.lead=rt.rank==0 or rt.log is None and rt.F is None          # the rank whose scheduler decides
        s.delta={};s.dsrc='none'
        if DT and os.path.exists(DT):
            d=json.load(open(DT));assert d.get('schema')=='nq-delta-v1',d.get('schema')
            for L in s.layers:s.delta[L]=torch.tensor(d['per_layer'][str(L)]['delta'],dtype=torch.float32,device=s.dev)
            s.dsrc=DT
        s.gamma=load_gamma(s.layers,s.dev) if GAMMA else {}
        s.rank_row={'count':0,'gate':1,'salience':3}[RANK]
        if RANK=='salience' and not s.delta:log.warning('NestQuant lookahead: no delta table, salience = sum w^2 ||x||^2')
        s.cid=0;s.q=collections.deque();s.pool=[torch.zeros(10,NE,dtype=torch.float32).pin_memory() for _ in range(4*len(s.layers)+8)];s.pi=0
        s.plan={};s.cs=collections.defaultdict(lambda:collections.defaultdict(float))
        s.evs=[]                                      # router-cost events (start, end) per chunk
        # measurement state
        s.pred={};s.prev={};s.acc=collections.defaultdict(float);s.nch=0
        if s.meas:
            fj=json.load(open(os.environ.get('NQ_HOME','/nq')+'/threads/22-boundary-experts/fixed_set.json'))
            s.static={L:torch.tensor(fj['n_routed'][str(L)],dtype=torch.float32,device=s.dev) for L in s.layers}
            s.fixed={L:torch.tensor(rt.S.fixed[rt.S.li[L]] if getattr(rt,'S',None) is not None else np.zeros(NE,bool),device=s.dev) for L in s.layers}
        miss=[L for L in s.layers if L not in _run or _run[L].gate is None]
        log.info('NestQuant prefill adapt: mode %s d %d budget %d rank score %s (delta table %s), measure %s, leader %s, '
                 'norm rescale %d layers, routers for %d/%d layers%s',MODE,D,BUDGET,RANK,s.dsrc,s.meas,s.lead,len(s.gamma),
                 len(s.layers)-len(miss),len(s.layers),
                 f' (missing {miss[:6]})' if miss else '')
    def xfor(s,L,Lt,x):
        gl,gt=s.gamma.get(L),s.gamma.get(Lt)
        if gl is None or gt is None or Lt==L:return x
        return (x.float()*(gt/gl.abs().clamp_min(1e-6)*gl.sign())).to(x.dtype)
    # ------------------------------------------------------------------ forward thread (prefill chunk, rank 0)
    def pre(s,L,x,ids,w,table):
        """called in forward after the mailbox apply, before M.prefill; x = the router's input (bf16 [T, H])"""
        if not s.lead:return
        if L==s.layers[0]:s.cid+=1;s.evs.append([])
        dl=s.delta.get(L);x2=x.float().pow(2).sum(1);A=stats(ids,w.float(),x2,dl)
        if s.meas:s._measure(L,x,x2,A)
        if not s.on:return
        buf=s.pool[s.pi];s.pi=(s.pi+1)%len(s.pool)
        e0=torch.cuda.Event(enable_timing=True);e1=torch.cuda.Event(enable_timing=True);e0.record()
        if MODE=='lookahead':
            Lt=L+D
            r=route(Lt,s.xfor(L,Lt,x)) if Lt in s.rt.lay else None
            P=stats(*r,x2,s.delta.get(Lt)) if r is not None else None
        else:Lt=L;P=A
        e1.record();s.evs[-1].append((e0,e1))
        st=torch.cat([A,table[:,0].float()[None],P if P is not None else torch.zeros(4,NE,device=x.device),
                      torch.zeros(1,NE,device=x.device)])
        buf.copy_(st,non_blocking=True);ev=torch.cuda.Event();ev.record()
        s.q.append((L,s.cid,Lt if P is not None else None,buf,ev))
        s.rt.wake.set()
    # ------------------------------------------------------------------ streaming thread (leader)
    def service(s,S,X,oplog):
        n=0
        while s.q and s.q[0][4].query():
            L,cid,Lt,buf,ev=s.q.popleft();a=buf.numpy();A=a[0:4];lvl=a[4];P=a[5:9];n+=1;c=s.cs[cid];i=S.li[L]
            # layer L served this chunk: its level-4 set and (if planned) the chosen set vs its actual routing
            tot=A.sum(1)+1e-30;l4=lvl==4
            for j,nm in enumerate(MEAS_N):c['served_'+nm]+=A[j][l4].sum()/tot[j]
            c['nl']+=1
            k=(L,cid)
            if k in s.plan:
                ups,chosen=s.plan.pop(k);c['landed']+=int(l4[ups].sum()) if len(ups) else 0
                for j,nm in enumerate(MEAS_N):c['chosen_'+nm]+=A[j][chosen].sum()/tot[j]
                c['np']+=1
            if Lt is not None:
                ups,downs,dfr,chosen=s._plan(S,Lt,P[s.rank_row])
                if ups or downs:
                    X.apply(ups,downs,S)
                    if oplog is not None:oplog.put(ups,downs)
                tc=cid if MODE=='lookahead' else cid+1
                s.plan[(Lt,tc)]=(np.array([e for _,e in ups],np.int64),chosen)
                s.cs[tc]['issued']+=len(ups);s.cs[tc]['deferred']+=dfr;s.cs[tc]['downs']+=len(downs)
            if L==s.layers[-1]:s._chunk_log(cid)
        return n
    def _plan(s,S,Lt,b):
        i=S.li[Lt];fx=S.fixed[i];st=S.state[i]
        sc=np.where(fx|(b<=0),-np.inf,b);o=np.argsort(-sc,kind='stable');top=[int(e) for e in o[:S.nf] if sc[e]>-np.inf]
        want=np.zeros(NE,bool);want[top]=True
        if len(top)<S.nf:                           # fill with experts already at (or on the way to) level 4: no churn
            keep=[int(e) for e in np.nonzero(np.isin(st,(1,2))&~want&~fx)[0]]
            keep.sort(key=lambda e:-S.score[i,e]);want[keep[:S.nf-len(top)]]=True
        cand=[e for e in top if st[e]==0 and S.hold[i,e]<=S.tok][:BUDGET]
        free=S.slots-int((S.state>0).sum()) if S.slots is not None else len(cand)
        ups=cand[:max(0,min(len(cand),free))]
        dfr=len(cand)-len(ups)+sum(1 for e in top[:BUDGET] if st[e]==3)
        downs=[int(e) for e in np.nonzero((st==2)&~want)[0]]
        S.want[i]=want
        for e in ups:S.state[i,e]=1
        for e in downs:S.state[i,e]=3
        S.stats['ups']+=len(ups);S.stats['downs']+=len(downs);S.stats['bytes']+=len(ups)*S.rb
        S.stats['la_ups']=S.stats.get('la_ups',0)+len(ups)
        return [(Lt,e) for e in ups],[(Lt,e) for e in downs],dfr,np.nonzero(fx|want)[0]
    def _chunk_log(s,cid):
        c=s.cs.pop(cid,None)
        if not c:return
        ms=0.
        if s.evs:
            ev=s.evs.pop(0)
            try:ms=sum(a.elapsed_time(b) for a,b in ev)
            except Exception:ms=-1
        nl=max(c['nl'],1);np_=max(c['np'],1)
        log.info('NestQuant prefill adapt chunk %d (%s d %d, rank %s): issued %d landed-in-time %d deferred %d downs %d; '
                 'chosen set share route %.3f gate %.3f sal %.3f dsal %.3f; served level-4 share route %.3f gate %.3f sal %.3f '
                 'dsal %.3f; router %.1f ms',cid,MODE,D,RANK,c['issued'],c['landed'],c['deferred'],c['downs'],
                 *(c['chosen_'+n]/np_ for n in MEAS_N),*(c['served_'+n]/nl for n in MEAS_N),ms)
    # ------------------------------------------------------------------ lookahead accuracy (NQ_LA_MEASURE=1)
    # definitions as the lead's prefill_ref (nq_prefill.py accuracy): fixed set excluded from the ranking; capture =
    # share of the chunk's actual measure on fixed + top-N(pred, non-fixed, positive score); recall = |top-N(pred) &
    # top-N(act)| / |top-N(act)|. Sources: d1/d2 (router of L+d on x_L, norm-rescaled), d1raw (no rescale), d0 (own
    # router on x_L: must equal the actual routing), prev (previous chunk of L), static (thread-22 n_routed), oracle.
    def _measure(s,L,x,x2,A):
        for d in (0,1,2):
            Lt=L+d
            if Lt in s.rt.lay:
                r=route(Lt,s.xfor(L,Lt,x))
                if r is not None:s.pred[(Lt,f'd{d}')]=stats(*r,x2,s.delta.get(Lt))
                if d==1 and s.gamma:
                    r=route(Lt,x)
                    if r is not None:s.pred[(Lt,'d1raw')]=stats(*r,x2,s.delta.get(Lt))
        src={k:s.pred.pop((L,k),None) for k in ('d0','d1','d2','d1raw')}
        src['prev']=s.prev.get(L);src['static']=s.static[L][None].expand(4,NE);src['oracle']=A;s.prev[L]=A
        names=[k for k,v in src.items() if v is not None];P=torch.stack([src[k] for k in names])      # [S, 4 rank, NE]
        fx=s.fixed[L]
        P=torch.where(fx[None,None],torch.full_like(P,-1.),P)
        v,idx=torch.sort(P,dim=2,descending=True,stable=True)
        Ao=torch.where(fx[None],torch.full_like(A,-1.),A);va,ia=torch.sort(Ao,dim=1,descending=True,stable=True)
        caps=[];recs=[]
        for N in KS:
            sel=torch.zeros_like(P,dtype=torch.bool).scatter_(2,idx[:,:,:N],v[:,:,:N]>0)            # [S, 4, NE] pred top-N
            act=torch.zeros_like(A,dtype=torch.bool).scatter_(1,ia[:,:N],va[:,:N]>0)                 # [4, NE] act top-N
            m=sel|fx[None,None]
            caps.append((m[:,:,None,:]*A[None,None]).sum(3))                                          # [S, rank, cap]
            recs.append((sel&act[None]).sum(2).float()/act.sum(1).clamp_min(1)[None].float())         # [S, rank]
        cap=torch.stack(caps,-1).cpu().numpy();rec=torch.stack(recs,-1).cpu().numpy();tot=A.sum(1).cpu().numpy()
        for si,nm in enumerate(names):
            for ci,cn in enumerate(MEAS_N):s.acc[(L,nm,'tot',cn)]+=float(tot[ci])
            s.acc[(L,nm,'n')]+=1
            for ri,rn in enumerate(MEAS_N):
                for ki,k in enumerate(KS):
                    s.acc[(L,nm,rn,'rec',k)]+=float(rec[si,ri,ki])
                    for ci,cn in enumerate(MEAS_N):s.acc[(L,nm,rn,cn,k)]+=float(cap[si,ri,ci,ki])
        if L==s.layers[-1]:s.nch+=1;s._dump()
    def _dump(s):
        bands={'L3-6':range(3,7),'L7-40':range(7,41),'L41-77':range(41,78),'all':range(3,78)}
        out=dict(chunks=s.nch,delta=s.dsrc,ks=KS,measures=MEAS_N,norm_rescale=bool(s.gamma),
                 note='bands[band][src][rank_by][captured or rec][N]: capture = sum(actual on fixed + top-N(pred)) / sum(actual), '
                      'recall = mean |top-N(pred) & top-N(act)| / |top-N(act)| (fixed excluded), over chunks x layers',bands={})
        srcs=sorted({k[1] for k in s.acc if len(k)==3 and k[2]=='n'})
        for bn,br in bands.items():
            Ls=[L for L in s.layers if L in br];ob={}
            for sn in srcs:
                t={cn:sum(s.acc.get((L,sn,'tot',cn),0.) for L in Ls) for cn in MEAS_N};n=sum(s.acc.get((L,sn,'n'),0) for L in Ls)
                ob[sn]={rn:dict({cn:{str(k):round(sum(s.acc.get((L,sn,rn,cn,k),0.) for L in Ls)/max(t[cn],1e-30),4) for k in KS} for cn in MEAS_N},
                                rec={str(k):round(sum(s.acc.get((L,sn,rn,'rec',k),0.) for L in Ls)/max(n,1),4) for k in KS}) for rn in MEAS_N}
                ob[sn]['n_chunk_layers']=n
            out['bands'][bn]=ob
        try:
            with open(OUT+'.tmp','w') as f:json.dump(out,f,indent=1)
            os.replace(OUT+'.tmp',OUT)
        except OSError as e:log.warning('NestQuant lookahead stats: %s',e)
        a=out['bands']['all'];g=lambda sn,rn,cn,k:a.get(sn,{}).get(rn,{}).get(cn,{}).get(str(k),float('nan'))
        log.info('NestQuant lookahead accuracy (%d chunks, all layers, N=45, capture by own measure): %s',s.nch,
                 ' '.join(f"{sn} route {g(sn,'count','count',45):.3f} gate {g(sn,'gate','gate',45):.3f} x2 {g(sn,'sal','sal',45):.3f} "
                          f"sal {g(sn,'dsal','dsal',45):.3f};" for sn in srcs))
