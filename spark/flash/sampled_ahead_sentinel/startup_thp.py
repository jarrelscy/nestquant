"""Process-local startup allocation diagnostic; no host policy writes."""
import atexit,ctypes,json,os,pathlib,time,functools
libc=ctypes.CDLL(None,use_errno=True)
libc.prctl.argtypes=[ctypes.c_int,ctypes.c_ulong,ctypes.c_ulong,ctypes.c_ulong,ctypes.c_ulong]
libc.prctl.restype=ctypes.c_int
SET,GET=41,42
def get():
 r=libc.prctl(GET,0,0,0,0)
 if r<0:raise OSError(ctypes.get_errno(),'PR_GET_THP_DISABLE')
 return r
def set_state(v):
 if libc.prctl(SET,int(v),0,0,0)<0:raise OSError(ctypes.get_errno(),'PR_SET_THP_DISABLE')
 if get()!=v:raise RuntimeError('THP state verification failed')
original=int(os.environ.setdefault('NQ_STARTUP_THP_ORIGINAL',str(get())))
if original not in (0,1):raise RuntimeError('Unsupported original THP state')
def record(phase):
 p=pathlib.Path(os.environ.get('NQ_STARTUP_THP_RECEIPTS','/data/kld-sampled-ahead-sentinel'))
 p.mkdir(parents=True,exist_ok=True)
 with (p/f'thp-{os.getpid()}.jsonl').open('a') as f:f.write(json.dumps({'pid':os.getpid(),'time':time.time(),'phase':phase,'original':original,'actual':get()})+'\n')
set_state(1);record('startup_disabled')
def restore():
 set_state(original);record('restored')
atexit.register(restore)
def install(loader):
 old=loader.patch_warmup
 def patch(module):
  old(module);wrapped=module.warmup_kernels
  @functools.wraps(wrapped)
  def warmup(*args,**kwargs):
   record('before_scheduler_warmup')
   try:return wrapped(*args,**kwargs)
   finally:restore()
  module.warmup_kernels=warmup
 loader.patch_warmup=patch
