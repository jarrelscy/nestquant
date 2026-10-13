import pathlib,json,hashlib,subprocess,sys,time,math
from huggingface_hub import hf_hub_download
P=pathlib.Path('/eval');revision='95f4fdd94bf29989db2e0d1054e4931f55edb6aa';manifest=json.loads((P/'full-panel-manifest.json').read_text());state={'stage':'downloading_teacher','completed':0,'total':4}
def save():state['updated_at']=time.time();(P/'scoring-status.json').write_text(json.dumps(state,indent=2))
save();results=[]
for i in range(4):
 tag=f'confirmation-{i:04d}';name=f'logits/full-panel/confirmation/{tag}.safetensors';state['window']=tag;save()
 f=P/'teacher'/name;assert f.is_file(), 'Pinned existing teacher artifact missing'
 h=hashlib.sha256()
 with f.open('rb') as fd:
  while b:=fd.read(8*1024*1024):h.update(b)
 assert h.hexdigest()==next(x['sha256'] for x in manifest['logit_files'] if x['path']==name)
 q=P/'output'/tag;state['stage']='scoring';save()
 subprocess.run([sys.executable,str(P/'score.py'),str(f),str(q/'decode.f32'),str(P/'tokens'/f'{tag}.tokens.npy'),str(q/'decode.json'),str(q/'score.json')],check=True)
 results.append(json.loads((q/'score.json').read_text()));state['completed']=i+1;save()
n=sum(x['positions'] for x in results)
assert len(results)==4 and n==8188 and all(x['positions']==2047 for x in results)
summary={'stage':'complete','positions':n,'kld':sum(x['kld']*x['positions'] for x in results)/n,'top1_agreement':sum(x['top1_agreement']*x['positions'] for x in results)/n,'student_ppl':math.exp(sum(math.log(x['student_ppl'])*x['positions'] for x in results)/n),'teacher_ppl':math.exp(sum(math.log(x['teacher_ppl'])*x['positions'] for x in results)/n),'windows':[{k:v for k,v in x.items() if k!='per_position_kld'} for x in results]};(P/'results-remote.json').write_text(json.dumps(summary,indent=2));state['stage']='complete';save();print(json.dumps(summary),flush=True)
