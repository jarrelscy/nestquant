"""Host wall-time accounting; no CUDA calls, no per-packet disk writes."""
import atexit,functools,json,os,pathlib,threading,time
_local=threading.local()
_totals={}
_request=0
_phase='startup'
_out=None

def timed(name,fn):
    @functools.wraps(fn)
    def wrapped(*args,**kwargs):
        stack=getattr(_local,'stack',None)
        if stack is None:stack=[];_local.stack=stack
        frame=[time.perf_counter_ns(),0]
        phase=_phase
        stack.append(frame)
        try:return fn(*args,**kwargs)
        finally:
            elapsed=time.perf_counter_ns()-frame[0]
            stack.pop()
            if stack:stack[-1][1]+=elapsed
            key=phase+':'+name
            row=_totals.setdefault(key,{'calls':0,'inclusive_ns':0,'exclusive_ns':0,'max_ns':0})
            row['calls']+=1;row['inclusive_ns']+=elapsed
            row['exclusive_ns']+=max(0,elapsed-frame[1]);row['max_ns']=max(row['max_ns'],elapsed)
    return wrapped

def dump(reason):
    if _out is None:return
    payload={'pid':os.getpid(),'request_resets':_request,'reason':reason,'time':time.time(),
             'scope':'Cumulative host wall time, includes existing GPU waits; nested inclusive times are not additive',
             'timings':_totals}
    path=_out/f'packet-timing-{os.getpid()}-{_request:03d}.json'
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(payload,indent=2));tmp.replace(path)

def install(runtime,jt,out):
    global _out
    if getattr(runtime.Runtime,'_packet_timing_installed',False):return
    _out=pathlib.Path(out);_out.mkdir(exist_ok=True)
    for name in ['begin','finish','sampled','poll','dispatch']:
        setattr(runtime.Runtime,name,timed('runtime.'+name,getattr(runtime.Runtime,name)))
    original_begin=runtime.Runtime.begin
    @functools.wraps(original_begin)
    def begin(self,*args,**kwargs):
        global _phase
        _phase='startup' if self.dummy else ('prefill' if self.batch.has_prefill else 'decode')
        return original_begin(self,*args,**kwargs)
    runtime.Runtime.begin=begin
    for name in ['prepare','commit']:
        setattr(jt.CommittedPredictor,name,timed('predictor.'+name,getattr(jt.CommittedPredictor,name)))
    jt.IncrementalJT.predict=timed('net.predict',jt.IncrementalJT.predict)
    # jt imports pickle module: proxy its dump only in this module, not globally.
    class PickleProxy:
        def __getattr__(self,name):return getattr(original_pickle,name)
    original_pickle=jt.pickle
    proxy=PickleProxy();proxy.dump=timed('trace.pickle',original_pickle.dump);jt.pickle=proxy
    reset=jt.CommittedPredictor.reset
    @functools.wraps(reset)
    def reset_wrapped(self,*args,**kwargs):
        global _request
        dump('before_predictor_reset');_request+=1
        return reset(self,*args,**kwargs)
    jt.CommittedPredictor.reset=reset_wrapped
    runtime.Runtime._packet_timing_installed=True
    atexit.register(dump,'atexit')
