"""Portable host orchestration of the byte-identical accepted Spark overlays."""
import argparse,hashlib,json,os,pathlib,shutil,subprocess,sys,time
HERE=pathlib.Path(__file__).resolve().parent
IMAGE='docker.io/jarrelscy/nestquant@sha256:3859a4e0018ec915ca0d4a3f9488158026fc2525534051433e2b37a729102364'
DATA=pathlib.Path(os.environ.get('NQ_FLASH_DATA',str(pathlib.Path.home()/'nq-flash'))).expanduser().resolve()
P=DATA/'kld-sampled-ahead-sentinel';NAME=os.environ.get('NQ_FLASH_CONTAINER','nq-spark-kld-sampled-ahead-sentinel')
def run(args,**kw):return subprocess.run(args,check=True,**kw)
def stage():
 P.mkdir(parents=True,exist_ok=True)
 manifest=json.loads((HERE/'tested-files.sha256.json').read_text())
 for name,want in manifest.items():
  assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==want,name
  target=P/name
  if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()==want:continue
  shutil.copyfile(HERE/name,target)
 for name in ('full-panel-manifest.json','teacher-hashes.json','tested-files.sha256.json','fetch_teacher.py'):
  shutil.copyfile(HERE/name,P/name)
 (P/'staged-receipt.json').write_text(json.dumps(dict(image=IMAGE,files=manifest),indent=2))
def stop():
 run(['docker','stop','-t','10',NAME])
 with (P/'serving.log').open('wb') as f:run(['docker','logs',NAME],stdout=f,stderr=subprocess.STDOUT)
 (P/'final-container-state.json').write_text(subprocess.check_output(['docker','inspect','--format','{{json .State}}',NAME],text=True))
def main():
 action=argparse.ArgumentParser();action.add_argument('action',choices=['stage','up','kld','fetch-teacher','score','down']);a=action.parse_args().action
 if a=='stage':stage();return
 if a=='down':stop();return
 if a=='score':
  assert not subprocess.check_output(['docker','ps','-q','--filter','name=^/'+NAME+'$'],text=True).strip(),'Stop serving before CPU scoring'
  run(['docker','run','--rm','--cpus','4','--memory','4g','-v',str(P)+':/eval','--entrypoint','python',IMAGE,'/eval/score_remote.py']);return
 if a=='fetch-teacher':
  stage();run(['docker','run','--rm','--cpus','4','--memory','4g','-v',str(P)+':/eval','-e','HF_TOKEN','--entrypoint','python',IMAGE,'/eval/fetch_teacher.py']);return
 if a=='up':
  assert os.environ.get('VLLM_API_KEY'),'Set VLLM_API_KEY privately in environment'
  assert not subprocess.check_output(['docker','ps','-q'],text=True).strip(),'Use an idle Spark; no other containers are stopped'
  assert not subprocess.check_output(['docker','ps','-aq','--filter','name=^/'+NAME+'$'],text=True).strip(),'Preserve/remove previous named container explicitly before starting'
  for n in ('status.json','sentinel-capture-gate.json','speed-result.json'):
   assert not (P/n).exists(),'Use a fresh data directory for a new diagnostic; preserve prior artifacts'
  stage()
  info=json.loads(subprocess.check_output(['docker','image','inspect',IMAGE,'--format','{{json .}}'],text=True));assert info['Architecture']=='arm64' and info['Os']=='linux'
  sys.path.insert(0,str(P));from resource_snapshot import snapshot
  (P/'prelaunch-resources.json').write_text(json.dumps(snapshot(),indent=2))
  env=os.environ.copy();env.update(NQ_FLASH_IMAGE=IMAGE,NQ_FLASH_DATA=str(DATA),NQ_FLASH_CONTAINER=NAME,NQ_GUARD_LOG_DIR=str(P/'guard'))
  with (P/'guard-supervisor.log').open('ab') as f:g=subprocess.Popen([sys.executable,str(P/'guard.py')],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  time.sleep(2);assert g.poll() is None,'Memory guard did not start'
  try:
   with (P/'launch.log').open('wb') as f:run(['bash',str(P/'run_spark_kld.sh'),'up'],env=env,stdout=f,stderr=subprocess.STDOUT)
  except BaseException:g.terminate();raise
  (P/'portable-launch-receipt.json').write_text(json.dumps(dict(image=IMAGE,container=NAME,guard_pid=g.pid,time=time.time()),indent=2));print('Started '+NAME+' with memory guard; inspect docker logs for readiness.');return
 if a=='kld':
  assert all((P/'tokens'/f'confirmation-{i:04d}.tokens.npy').is_file() for i in range(4)),'Run fetch-teacher first'
  try:
   with (P/'client.log').open('wb') as f:run(['docker','exec',NAME,'python','/data/kld-sampled-ahead-sentinel/client.py'],stdout=f,stderr=subprocess.STDOUT,timeout=3600)
  finally:stop()
  print('Capture stopped safely. Run score on the idle Spark or copy this directory to a CPU host.')
if __name__=='__main__':main()
