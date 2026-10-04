"""NQ_LMPF (windowed layer-major 4-bit prefill), EngineCore side. Loaded by sitecustomize.py when NQ_LMPF=1 (chained
from nq_pb_engine.patch_scheduler when NQ_PREFILL_BORROW=1, so this wrapper is the INNER one and prefill-borrow sees
out.nq_lmpf). Patches vllm.v1.core.sched.scheduler.Scheduler.schedule.

Per schedule() call:
  pre   when the step will prefill exactly one text request with more than max_num_batched_tokens (mnbt) tokens left
        and at least NQ_LMPF_BUD_MIN of them: grant a window of w tokens by setting max_num_scheduled_tokens = w for
        this call (nwin = ceil(left / W), w = ceil(left / nwin), W = min(NQ_LMPF_WINDOW, every rank's ready-file W)) and
        max_num_running_reqs = max(1, len(running)) (no other request is admitted next to it). Restored after the call.
  post  annotates out.nq_lmpf = dict(m, rid, start, n, last, new, end, budget_s, mnbt) when one request is scheduled and it
        is prefilling: m = full (its uncached new tokens >= NQ_LMPF_MIN_NEW), bud (>= NQ_LMPF_BUD_MIN, budget > 0),
        plain (neither, but n > mnbt: the worker runs it as normal sub-chunks). Mode, new and budget are fixed at the
        request's first prefill step (new = prompt - cached). Otherwise out.nq_lmpf = None.
Every rank gets the same annotation, so every TP/DCP rank runs the same order.
Workers write /dev/shm/nq_lmpf_ready_<enginepid>_<starttime>_r<rank>.json (W, C, mnbt) once their ring and buffers fit;
until all ranks have, nothing is granted or annotated (bud single-step prompts need it too: all ranks agree).
Runtime knobs (re-read when the file's mtime changes): /dev/shm/nq_lmpf_min_new, /dev/shm/nq_lmpf_bud_min,
/dev/shm/nq_lmpf_budget_s; /dev/shm/nq_lmpf_off = off (no new grants or annotations)."""
import os,sys,json,time,logging,functools
ON=os.environ.get('NQ_LMPF','0')=='1'
OFF=os.environ.get('NQ_LMPF_OFF','/dev/shm/nq_lmpf_off')
WINDOW=int(os.environ.get('NQ_LMPF_WINDOW','32768'))
log=logging.getLogger('vllm.nestquant.lmpf')   # child of vllm's logger; no vllm import at site time

class Knob:
    """env default, overridden by the first token of a /dev/shm file while it exists (re-parsed on mtime change)"""
    def __init__(s,env,dflt,path,typ):
        s.typ=typ;s.path=path;s.mt=None;s.fv=None
        try:s.dflt=typ(os.environ.get(env,'') or dflt)
        except ValueError:s.dflt=typ(dflt)
    def __call__(s):
        try:m=os.stat(s.path).st_mtime_ns
        except OSError:return s.dflt
        if m!=s.mt:
            try:s.fv=s.typ(open(s.path).read().split()[0]);s.mt=m
            except Exception:return s.dflt
        return s.fv
MIN_NEW=Knob('NQ_LMPF_MIN_NEW',32768,'/dev/shm/nq_lmpf_min_new',int)
BUD_MIN=Knob('NQ_LMPF_BUD_MIN',1024,'/dev/shm/nq_lmpf_bud_min',int)
BUDGET_S=Knob('NQ_LMPF_BUDGET_S',2.0,'/dev/shm/nq_lmpf_budget_s',float)

def boot_key(pid=None):
    """/dev/shm key of this boot: the engine-core pid + its start time (workers: their parent's), so a stale file of a
    previous boot with a reused pid never matches"""
    pid=os.getpid() if pid is None else pid
    st=open(f'/proc/{pid}/stat').read().rsplit(')',1)[1].split()[19]
    return f'{pid}_{st}'
def ready_path(key,rank):return f'/dev/shm/nq_lmpf_ready_{key}_r{rank}.json'

def windows(left,W,q=1):
    """(nwin, w): left tokens in nwin windows of <= W tokens, as even as possible, w rounded up to a multiple of q
    (q = mnbt: every window then splits into full mnbt chunks, so only the prompt's last chunk is short, as in prod
    chunked prefill; short chunks miss VLLM_GLM_RAW_KV_GATHER and take the per-layer NCCL path, ~3 s per deep window)"""
    nwin=max(1,-(-left//max(1,W)));w=-(-left//nwin)
    if q>1 and W%q==0:w=min(W,-(-w//q)*q)
    return nwin,w

def decide(new,min_new,bud_min,budget_s):
    """mode of a prefill with `new` uncached prompt tokens"""
    if new>=min_new:return 'full'
    if new>=bud_min and budget_s>0:return 'bud'
    return None

class State:
    def __init__(s,sched,key=None):
        s.dead=False;s.req={};s.saved=None;s.n=dict(grants=0,full=0,bud=0,plain=0,steps=0);s.warned=0
        s.mnbt=int(sched.max_num_scheduled_tokens)
        try:s.world=int(sched.vllm_config.parallel_config.world_size)
        except Exception:s.world=1
        try:s.key=key or boot_key()
        except Exception as e:log.warning('NestQuant LMPF: no boot key (%s), off',e);s.dead=True;s.key=''
        s.rd=None;s.rd_t=0.
        log.info('NestQuant LMPF: scheduler hook on (mnbt %d, world %d, window %d, min_new %d, bud_min %d, budget %.2f s, key %s)',
                 s.mnbt,s.world,WINDOW,MIN_NEW(),BUD_MIN(),BUDGET_S(),s.key)
    def ready(s):
        """min W over all ranks' ready files, or None while any rank is missing (re-checked at most every 2 s until all
        are there, then every 30 s: a rank that turns LMPF off removes its file)"""
        t=time.time()
        if s.rd is not None and t-s.rd_t<30.:return s.rd
        if s.rd is None and t-s.rd_t<2.:return None
        s.rd_t=t;W=WINDOW
        for r in range(s.world):
            try:d=json.load(open(ready_path(s.key,r)))
            except (OSError,ValueError):s.rd=None;return None
            W=min(W,int(d.get('W',W)))
        s.rd=W;return W
    def candidate(s,sched):
        """(request, tokens left) the next schedule() will prefill alone, or None"""
        rs=sched.running
        if len(rs)>1:return None
        if rs:
            r=rs[0]
            if r.num_computed_tokens>=r.num_prompt_tokens:return None
        else:
            if not (sched.waiting or getattr(sched,'skipped_waiting',None)):return None
            try:q=sched._select_waiting_queue_for_scheduling()
            except Exception:q=sched.waiting or None
            if q is None:return None
            r=q.peek_request()
            if getattr(r.status,'name','') not in ('WAITING','PREEMPTED'):return None
        if r.has_encoder_inputs or getattr(r,'lora_request',None) is not None:return None
        if getattr(r,'pooling_params',None) is not None:return None
        return r,r.num_tokens-r.num_computed_tokens
    def pre(s,sched):
        s.saved=None
        if s.dead or not ON or os.path.exists(OFF):return
        c=s.candidate(sched)
        if c is None:return
        r,left=c
        if left<=s.mnbt or left<BUD_MIN():return
        st=s.req.get(r.request_id)
        if st is not None and st['m'] not in ('full','bud'):return
        if st is None and decide(r.num_tokens-r.num_computed_tokens,MIN_NEW(),BUD_MIN(),BUDGET_S()) is None:return
        W=s.ready()
        if W is None:
            if s.warned<3:s.warned+=1;log.warning('NestQuant LMPF: worker ready files missing (%s), no window',s.key)
            return
        _,w=windows(left,max(W,s.mnbt),s.mnbt)
        if w<=s.mnbt:return
        s.saved=(sched.max_num_scheduled_tokens,sched.max_num_running_reqs)
        sched.max_num_scheduled_tokens=w;sched.max_num_running_reqs=max(1,len(sched.running));s.n['grants']+=1
    def restore(s,sched):
        if s.saved is not None:sched.max_num_scheduled_tokens,sched.max_num_running_reqs=s.saved;s.saved=None
    def post(s,sched,out):
        out.nq_lmpf=None
        nst=out.num_scheduled_tokens
        if len(nst)!=1:return
        (rid,ns),=nst.items()
        if ns<=0:return
        r=sched.requests.get(rid)
        if r is None:return
        before=r.num_computed_tokens-ns
        if before>=r.num_prompt_tokens:return
        if out.scheduled_spec_decode_tokens:return
        st=s.req.get(rid)
        if st is None:
            if s.dead or not ON or os.path.exists(OFF) or r.has_encoder_inputs or s.ready() is None:m=None
            else:m=decide(r.num_prompt_tokens-before,MIN_NEW(),BUD_MIN(),BUDGET_S())
            st=s.req[rid]=dict(m=m,new=r.num_prompt_tokens-before,b=BUDGET_S(),t=time.time())
            if len(s.req)>256:
                for k in sorted(s.req,key=lambda k:s.req[k]['t'])[:128]:s.req.pop(k,None)
        m=st['m']
        if m is None or r.has_encoder_inputs:
            if ns<=s.mnbt:return
            m='plain'
        s.n['steps']+=1;s.n[m]+=1
        out.nq_lmpf=dict(m=m,rid=rid,start=before,n=ns,last=before+ns>=r.num_prompt_tokens,new=st['new'],end=r.num_prompt_tokens,budget_s=st['b'],mnbt=s.mnbt)

_ST={}
def _state(sched):
    st=_ST.get(id(sched))
    if st is None:st=_ST[id(sched)]=State(sched)
    return st

def patch_scheduler(mod):
    C=mod.Scheduler
    if getattr(C,'_nq_lmpf',False):return
    s0=C.schedule
    @functools.wraps(s0)
    def schedule(self,*a,**k):
        st=None
        try:st=_state(self);st.pre(self)
        except Exception:
            log.exception('NestQuant LMPF: scheduler pre hook failed, off')
            if st is not None:st.restore(self);st.dead=True
        try:out=s0(self,*a,**k)
        finally:
            if st is not None:st.restore(self)
        try:
            if st is not None:st.post(self,out)
            else:out.nq_lmpf=None
        except Exception:
            log.exception('NestQuant LMPF: scheduler post hook failed, off')
            if st is not None:st.dead=True
            n=max(out.num_scheduled_tokens.values(),default=0)
            out.nq_lmpf=dict(m='plain',rid=next(iter(out.num_scheduled_tokens)),start=-1,n=n,last=False,new=0,end=0,budget_s=0.,mnbt=st.mnbt if st else n) \
                if st is not None and n>st.mnbt and len(out.num_scheduled_tokens)==1 else None
        return out
    C.schedule=schedule;C._nq_lmpf=True
    log.info('NestQuant LMPF: Scheduler.schedule patched')

_TARGETS={'vllm.v1.core.sched.scheduler':patch_scheduler}

class _Finder:
    """meta-path hook: run the patch right after the target module executes (no early vllm import)"""
    def find_spec(self,name,path=None,target=None):
        if name not in _TARGETS:return None
        for f in sys.meta_path:
            if f is self or not hasattr(f,'find_spec'):continue
            spec=f.find_spec(name,path,target)
            if spec is not None:break
        else:return None
        ld=spec.loader;ex0=ld.exec_module
        def exec_module(m,_ex0=ex0,_n=name):
            _ex0(m)
            try:_TARGETS[_n](m)
            except Exception:log.exception('NestQuant LMPF: patch of %s failed',_n)
        try:ld.exec_module=exec_module
        except Exception:return None
        return spec

def install():
    """standalone install (prefill-borrow off); with NQ_PREFILL_BORROW=1 nq_pb_engine.patch_scheduler chains us"""
    if not ON:return
    for n,f in _TARGETS.items():
        if n in sys.modules:f(sys.modules[n])
    if not any(isinstance(f,_Finder) for f in sys.meta_path):sys.meta_path.insert(0,_Finder())
