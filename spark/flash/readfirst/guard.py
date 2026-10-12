import pathlib,json,time,subprocess,os
p=pathlib.Path(os.environ.get('NQ_GUARD_LOG_DIR', str(pathlib.Path(__file__).resolve().parent)))
p.mkdir(parents=True, exist_ok=True)
name=os.environ.get('NQ_FLASH_CONTAINER', 'nq-spark-flash')
low=0;high=0;seen=False;low_since=None;high_since=None
for i in range(172800):
 row={'time':time.time(),'physical_source':'host ZFS arcstats; namespace PSI separately labeled'}
 try:
  vals={x.split()[0]:int(x.split()[2]) for x in pathlib.Path('/proc/spl/kstat/zfs/arcstats').read_text().splitlines()[2:] if len(x.split())==3}
  row['host']={k:vals[k] for k in ['memory_all_bytes','memory_free_bytes','memory_available_bytes','size']}
  row['namespace_memory_pressure']=pathlib.Path('/sys/fs/cgroup/memory.pressure').read_text().strip()
  row['namespace_memory_stat']={a:int(b) for a,b in (x.split() for x in pathlib.Path('/sys/fs/cgroup/memory.stat').read_text().splitlines()) if a in ['anon','file','shmem','pgscan','pgsteal','pgmajfault','unevictable']}
  full=float(next(x for x in row['namespace_memory_pressure'].splitlines() if x.startswith('full ')).split('avg10=')[1].split()[0])
  if full>10 and i%3==0:
   stacks=[]
   for proc in pathlib.Path('/proc').glob('[0-9]*'):
    try:
     comm=(proc/'comm').read_text().strip()
     if not any(k in comm for k in ['VLLM','python','pt_main_thread']):continue
     for task in (proc/'task').glob('*'):
      try:stacks.append(dict(pid=int(proc.name),tid=int(task.name),comm=(task/'comm').read_text().strip(),wchan=(task/'wchan').read_text().strip(),stack=(task/'stack').read_text()[:6000]))
      except OSError:pass
    except OSError:pass
   with (p/'pressure-stacks.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),full_psi=full,stacks=stacks))+'\n')
  low=low+1 if vals['memory_available_bytes']<5*1024**3 else 0
  high=high+1 if full>50 else 0
  now=time.monotonic()
  low_since=(low_since if low_since is not None else now) if low else None
  high_since=(high_since if high_since is not None else now) if high else None
  low_seconds=now-low_since if low_since is not None else 0
  high_seconds=now-high_since if high_since is not None else 0
  row.update(low_headroom_samples=low,high_pressure_samples=high,low_seconds=low_seconds,high_seconds=high_seconds)
  with (p/'host-memory.jsonl').open('a') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
  if vals['memory_available_bytes']<4*1024**3 or low_seconds>=15 or high_seconds>=15:
   row['boot_id']=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
   row['buddyinfo']=pathlib.Path('/proc/buddyinfo').read_text()
   row['vmstat']={a:int(b) for a,b in (x.split() for x in pathlib.Path('/proc/vmstat').read_text().splitlines()) if a.startswith(('compact','allocstall','oom'))}
   (p/'host-memory-guard.json').write_text(json.dumps(dict(row,action='stop diagnostic',reason='immediate below4GiB; below5GiB or namespace FULL PSI above50percent continuously for15seconds'),indent=2))
   try:subprocess.run(['docker','kill',name] if vals['memory_available_bytes']<4*1024**3 else ['docker','stop','-t','2',name],timeout=8,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
   except subprocess.TimeoutExpired:
    subprocess.run(['docker','kill',name],timeout=15,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
   break
  try:
   if i%5:time.sleep(1);continue
   state=subprocess.check_output(['docker','inspect','--format','{{.State.Status}}',name],text=True,timeout=3,stderr=subprocess.DEVNULL).strip()
   seen=True
   if state=='exited':break
  except Exception:
   if i>30 and not seen:break
 except Exception as e:
  with (p/'guard-errors.log').open('a') as f:f.write(type(e).__name__+'\n')
 time.sleep(1)
