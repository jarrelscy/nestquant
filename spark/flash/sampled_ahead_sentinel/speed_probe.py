"""Real sampled generation, exact server token usage; excludes prefill via metrics."""
import urllib.request,json,os,time,pathlib,re
P=pathlib.Path('/data/kld-sampled-ahead-sentinel')
def getmetrics():
 req=urllib.request.Request('http://127.0.0.1:8000/metrics',headers={'Authorization':'Bearer '+os.environ['VLLM_API_KEY']})
 text=urllib.request.urlopen(req,timeout=30).read().decode();out={}
 for name in ['vllm:request_decode_time_seconds_sum','vllm:request_decode_time_seconds_count']:
  vals=[float(l.split()[-1]) for l in text.splitlines() if l.startswith(name+'{') or l.startswith(name+' ')];out[name]=sum(vals)
 return out

def run(token):
 assert not (P/'force').exists(),'Teacher forcing must be absent'
 before=getmetrics();payload={'model':'glm-5.3-flash-nq','prompt':[int(token)],'max_tokens':512,'temperature':0.7,'top_p':0.95,'ignore_eos':True,'seed':1234,'stream':True,'stream_options':{'include_usage':True}}
 req=urllib.request.Request('http://127.0.0.1:8000/v1/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer '+os.environ['VLLM_API_KEY']})
 chunks=[];generated=[];usage=None;started=time.monotonic()
 with urllib.request.urlopen(req,timeout=1800) as r:
  for line in r:
   if not line.startswith(b'data: '):continue
   body=line[6:].strip()
   if body==b'[DONE]':break
   d=json.loads(body);now=time.monotonic()
   if d.get('usage'):usage=d['usage']
   for c in d.get('choices',[]):
    generated.append(c.get('text',''))
    ts=(c.get('logprobs') or {}).get('tokens') or []
    if ts:chunks.append({'time':now-started,'tokens':len(ts)})
 after=getmetrics();assert usage and usage['completion_tokens']==512
 metric_seconds=after['vllm:request_decode_time_seconds_sum']-before['vllm:request_decode_time_seconds_sum'];metric_count=after['vllm:request_decode_time_seconds_count']-before['vllm:request_decode_time_seconds_count'];assert metric_count==1 and metric_seconds>0,(before,after)
 result={'method':'real .7/.95 sampled generation; force absent; engine request decode histogram delta excludes prefill; (completion_tokens-1)/decode_seconds','prompt_token_ids':[int(token)],'generated_text':''.join(generated),'logprobs_requested':False,'usage':usage,'decode_seconds':metric_seconds,'decode_tps':(usage['completion_tokens']-1)/metric_seconds,'metric_before':before,'metric_after':after,'elapsed_seconds':time.monotonic()-started,'stream_chunks':chunks,'stream_token_count':sum(c['tokens'] for c in chunks),'teacher_forcing':False,'claim_scope':'single 512-token real generation probe, not full benchmark'}
 (P/'speed-result.json').write_text(json.dumps(result,indent=2));return result
