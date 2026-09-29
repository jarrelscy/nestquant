"""Teacher-forced prompt logprobs from a live vLLM serve, for comparison with eval_fp8's per-token dump (_tok.npz).
  serve_logprobs.py CORPUS.npy OUT.npz [label] [k=5]
Each window is sent as a token-id prompt (no BOS added, same as eval_fp8) with max_tokens=1, prompt_logprobs=k.
Saved per window: lp[p] = logprob of token p+1 given tokens <= p (same indexing as eval_fp8's nll), top1[p] and the
top-k ids/logprobs (a truncated distribution: vLLM returns only k entries)."""
import sys,json,re,time,urllib.request,numpy as np
cp,out=sys.argv[1],sys.argv[2];label=sys.argv[3] if len(sys.argv)>3 else '';k=int(sys.argv[4]) if len(sys.argv)>4 else 5
key=re.search(r'VLLM_API_KEY=(\S+)',open('/home/jarrelscy/homeassistant/.env').read()).group(1)
W=np.load(cp);res=dict(lp=[],top1=[],topk_ids=[],topk_lp=[])
for w in W:
    body=dict(model='local',prompt=[int(x) for x in w],max_tokens=1,temperature=0,prompt_logprobs=k)
    rq=urllib.request.Request('http://localhost:8001/v1/completions',json.dumps(body).encode(),{'Content-Type':'application/json','Authorization':'Bearer '+key})
    t=time.time();r=json.load(urllib.request.urlopen(rq,timeout=600));pl=r['choices'][0]['prompt_logprobs']
    assert pl[0] is None and len(pl)==len(w),(len(pl),len(w))
    lp=np.full(len(w)-1,np.nan);t1=np.full(len(w)-1,-1);ti=np.full((len(w)-1,k),-1);tl=np.full((len(w)-1,k),np.nan)
    for p in range(1,len(w)):
        d=pl[p];lp[p-1]=d[str(int(w[p]))]['logprob']
        o=sorted(((int(t),v['logprob']) for t,v in d.items() if v.get('rank',99)<=k),key=lambda x:-x[1])[:k]
        t1[p-1]=o[0][0];ti[p-1,:len(o)]=[x[0] for x in o];tl[p-1,:len(o)]=[x[1] for x in o]
    for a,b in zip(('lp','top1','topk_ids','topk_lp'),(lp,t1,ti,tl)):res[a].append(b)
    print(f'{label} window: mean nll {-lp.mean():.4f} ({time.time()-t:.1f}s)',flush=True)
np.savez_compressed(out,**{a:np.stack(v) for a,v in res.items()},label=label)
