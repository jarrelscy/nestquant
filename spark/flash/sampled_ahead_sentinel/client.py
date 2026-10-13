import os,json,pathlib,time,urllib.request,numpy as np
P=pathlib.Path('/data/kld-sampled-ahead-sentinel');state={'stage':'waiting_for_serving','completed':0,'total':4}
def save():state['updated_at']=time.time();(P/'status.json').write_text(json.dumps(state,indent=2))
def request(path,payload=None):
 data=None if payload is None else json.dumps(payload).encode();req=urllib.request.Request('http://127.0.0.1:8000'+path,data=data,headers={'Content-Type':'application/json','Authorization':'Bearer '+os.environ['VLLM_API_KEY']});return json.load(urllib.request.urlopen(req,timeout=1800))
save()
try:
 for _ in range(180):
  try:request('/v1/models');break
  except Exception:time.sleep(5)
 else:raise RuntimeError('Serving readiness deadline exceeded')
 receipts=[]
 for f in P.glob('thp-*.jsonl'):
  rows=[json.loads(v) for v in f.read_text().splitlines()]
  if any(v['phase']=='before_scheduler_warmup' for v in rows):
   assert rows[-1]['phase']=='restored' and rows[-1]['actual']==rows[-1]['original']==0,rows
   receipts.append(rows)
 assert receipts,'No Engine warmup restoration receipt; fail closed before capture'
 (P/'thp-capture-gate.json').write_text(json.dumps(receipts,indent=2))
 prewarm=json.loads((P/'mla-prewarm.json').read_text());assert prewarm['phase']=='before_hot_pool_after_KV_reservation' and prewarm['tokens']==1024 and prewarm['real_profile_retained']
 assert prewarm['completed_at']<min(v['time'] for rows in receipts for v in rows if v['phase']=='before_scheduler_warmup')
 sentinel_receipts=[]
 for f in P.glob('sentinel-*.json'):
  receipt=json.loads(f.read_text());events=receipt['events'];alloc=[x for x in events if x['event']=='allocated'];reuse=[x for x in events if x['event']=='reused'];release=[x for x in events if x['event']=='released']
  assert receipt['released'] and len(alloc)==1 and reuse and len(release)==1 and release[0]['accounting_corrected']
  assert all(x['pointer']==alloc[0]['pointer'] and x['bytes']==512*1024*1024 for x in reuse)
  assert alloc[0]['time']<prewarm['completed_at']<max(x['time'] for x in reuse)<release[0]['time']
  sentinel_receipts.append(receipt)
 assert sentinel_receipts,'No retained/reused/released startup sentinel receipt'
 (P/'sentinel-capture-gate.json').write_text(json.dumps(sentinel_receipts,indent=2))
 state['stage']='running';save()
 for i in range(4):
  tag=f'confirmation-{i:04d}';tokens=np.load(P/'tokens'/f'{tag}.tokens.npy').astype('int64').tolist();state['window']=tag;save()
  (P/'force').write_text(f'{tag} /data/kld-sampled-ahead-sentinel/tokens/{tag}.tokens.npy')
  start=time.monotonic()
  r=request('/v1/completions',{'model':'glm-5.3-flash-nq','prompt':[tokens[0]],'max_tokens':len(tokens)-1,'temperature':0.7,'top_p':0.95,'ignore_eos':True,'seed':1234})
  elapsed=time.monotonic()-start;(P/f'{tag}-response.json').write_text(json.dumps(r,indent=2));(P/'force').unlink()
  request('/v1/completions',{'model':'glm-5.3-flash-nq','prompt':[tokens[0]],'max_tokens':1,'temperature':0.7,'top_p':0.95,'ignore_eos':True})
  m=json.loads((P/'output'/tag/'decode.json').read_text());assert m['positions']==2047 and m['dup']==0 and m['first_mismatch']==0 and not m['missing'],m
  state['completed']=i+1;state['last_elapsed_seconds']=elapsed;state['last_usage']=r.get('usage');save()
 state['stage']='real_generation_speed';save()
 import speed_probe
 speed_probe.run(int(np.load(P/'tokens/confirmation-0000.tokens.npy')[0]))
 request('/v1/completions',{'model':'glm-5.3-flash-nq','prompt':[tokens[0]],'max_tokens':1,'temperature':0.7,'top_p':0.95,'ignore_eos':True})
 state['stage']='capture_complete';save()
except Exception as e:
 state['stage']='failed';state['error']=str(e);save();raise
finally:
 (P/'force').unlink(missing_ok=True)
