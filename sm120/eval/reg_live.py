"""Live decode KLD on the registry panel panel--glm53.malaiwah.corpus5x5-v1, measured on the running server.

Needs the server started with NQ_KLD_HOOK=1 (sm120/serve/nq_kld.py; compose mounts host NQ_KLD dumps at /dbg). Per context:
a 1-token prompt (token 0) + 2047 teacher-forced decode tokens with MTP on, so expert streaming, prefetch and verify
shapes are those of real decode. The hook records the full-vocab fp32 log_softmax of every committed row.
Reference = panel capture hidden_states.float() @ head lm_head.float().T (no norm re-applied); KL(ref||live) per position,
float64, full vocab, token mean over 25 x 2047 = 51,175 positions. Contexts run back to back (warm server, cold predictor).

  python reg_live.py run   --panel PANEL [--url http://localhost:8001] [--tag reg24live]   # OPENAI_API_KEY = server key
  python reg_live.py score --panel PANEL [--dbg /data/Jarrel/nq-serve/dbg/kld] [--tag reg24live] [--out res.json]
The decode.pt dumps are ~1.3 GB per context and root-owned; remove them via the container (rm -rf /dbg/kld/<tag>_c*)."""
import argparse,json,os,time,urllib.request,numpy as np

def run(a):
    FK='/dev/shm/nq_kld_force';key=os.environ['OPENAI_API_KEY']
    def post(d):
        r=urllib.request.Request(a.url+'/v1/completions',data=json.dumps(d).encode(),
                                 headers={'Content-Type':'application/json','Authorization':'Bearer '+key})
        return json.loads(urllib.request.urlopen(r,timeout=3600).read())
    for c in range(25):
        tag=f'{a.tag}_c{c:02d}'
        if os.path.exists(f'{a.dbg}/{tag}/decode.pt'):continue
        tk=np.asarray(json.load(open(f'{a.panel}/panel/tokens/context-{c:04d}.json')),np.int64);tp=f'/dev/shm/nq_regtok_{c}.npy';np.save(tp,tk)
        open(FK,'w').write(f'{tag} {tp}');t=time.time()
        try:post(dict(model='local',prompt=[int(tk[0])],max_tokens=2047,temperature=0,ignore_eos=True))
        finally:os.remove(FK)
        dt=time.time()-t
        post(dict(model='local',prompt=[int(tk[0])],max_tokens=1))   # knob gone -> hook flushes decode.pt
        for _ in range(60):
            if os.path.exists(f'{a.dbg}/{tag}/decode.pt'):break
            time.sleep(1)
        os.remove(tp);print(json.dumps(dict(ctx=c,s=round(dt,1),tps=round(2047/dt,1))),flush=True)

def score(a):
    import torch
    from safetensors import safe_open
    with safe_open(f'{a.panel}/head/weight.safetensors','pt') as f:W=f.get_tensor('lm_head.weight').float()
    N=2047;kl=[];agree=0;per={}
    for c in range(25):
        D=torch.load(f'{a.dbg}/{a.tag}_c{c:02d}/decode.pt');pos=D['pos']
        assert torch.equal(pos,torch.arange(N)) and D['dup']==0 and D['first_mismatch']==0,(c,pos.numel(),D['dup'],D['first_mismatch'])
        with safe_open(f'{a.panel}/capture/hidden_{c:04d}.safetensors','pt') as f:H=f.get_tensor('hidden_states').float()
        k=torch.empty(N,dtype=torch.float64);ag=0
        for i in range(0,N,256):
            j=min(N,i+256);lt=torch.log_softmax((H[i:j]@W.T).double(),-1)
            ls=torch.log_softmax(D['lp'][i:j,:W.shape[0]].double(),-1)
            k[i:j]=(lt.exp()*(lt-ls)).sum(-1);ag+=int((lt.argmax(-1)==ls.argmax(-1)).sum())
        kl.append(k);agree+=ag;per[c]=dict(kld=k.mean().item(),top1=ag/N,emitted_per_step=N/D['steps'])
        print(json.dumps(dict(ctx=c,**per[c])),flush=True)
    kl=torch.cat(kl);out=dict(panel='panel--glm53.malaiwah.corpus5x5-v1',n_positions=kl.numel(),kld=kl.mean().item(),
                              top1=agree/kl.numel(),kld_p50=kl.median().item(),kld_p99=kl.quantile(.99).item(),per_context=per)
    json.dump(out,open(a.out,'w'),indent=1);print(json.dumps({k:v for k,v in out.items() if k!='per_context'}))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('cmd',choices=['run','score']);ap.add_argument('--panel',required=True)
    ap.add_argument('--url',default='http://localhost:8001');ap.add_argument('--tag',default='reg24live')
    ap.add_argument('--dbg',default='/data/Jarrel/nq-serve/dbg/kld');ap.add_argument('--out',default='reg_live_results.json')
    a=ap.parse_args();(run if a.cmd=='run' else score)(a)
