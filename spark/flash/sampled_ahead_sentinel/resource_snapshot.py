"""Read-only namespace/host counters; deltas scoped to supplied baseline."""
import pathlib,time,os

def snapshot(baseline=None,cgroup=pathlib.Path('/sys/fs/cgroup'),proc=pathlib.Path('/proc')):
 out={'time':time.time(),'monotonic':time.monotonic(),'page_size_bytes':os.sysconf('SC_PAGE_SIZE'),'scope':'namespace cgroup counters; host vmstat/buddyinfo; peak is lifetime, not reset'}
 for name in ['memory.current','memory.peak','memory.high','memory.max','memory.swap.current','memory.swap.high','memory.swap.max','memory.swap.peak']:
  try:
   s=(cgroup/name).read_text().strip();out[name]=int(s) if s!='max' else s
  except OSError as e:out[name]={'unavailable':type(e).__name__}
 for name in ['memory.events','memory.events.local','memory.swap.events','memory.stat']:
  try:out[name]={a:int(b) for a,b in (l.split() for l in (cgroup/name).read_text().splitlines())}
  except OSError as e:out[name]={'unavailable':type(e).__name__}
 out['vmstat']={a:int(b) for a,b in (l.split() for l in (proc/'vmstat').read_text().splitlines()) if a.startswith(('compact','allocstall','oom','pgscan','pgsteal','pgmajfault','pswp'))}
 out['processes']=[]
 for d in proc.glob('[0-9]*'):
  try:
   comm=(d/'comm').read_text().strip()
   if not any(x in comm.lower() for x in ['python','vllm','pt_main','compile','triton']):continue
   status={a:b.strip() for a,b in (l.split(':',1) for l in (d/'status').read_text().splitlines() if ':' in l) if a in ['State','VmRSS','RssAnon','RssFile','RssShmem','Threads']}
   out['processes'].append({'pid':int(d.name),'comm':comm,'status':status,'wchan':(d/'wchan').read_text().strip()})
  except OSError:pass
 out['buddyinfo']=(proc/'buddyinfo').read_text()
 out['boot_id']=(proc/'sys/kernel/random/boot_id').read_text().strip()
 if baseline is not None:
  if out['boot_id']!=baseline['boot_id']:raise RuntimeError('Counter baseline belongs to different boot')
  out['baseline_monotonic']=baseline['monotonic'];out['counter_deltas']={}
  for key in ['memory.events','memory.events.local','memory.swap.events','vmstat']:
   out['counter_deltas'][key]={k:v-baseline[key][k] for k,v in out[key].items() if isinstance(v,int) and isinstance(baseline[key].get(k),int)}
 return out
