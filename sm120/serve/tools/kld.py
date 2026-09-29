"""Teacher-forced quality probe on the served model (C2). No FP8 reference fits on this box, so we report:
  - ppl of the real next token (reference-free), and
  - top-20 KLD / top-1 agreement between two served configs (e.g. NQ dynamic vs ARVQ hybrid, run vs repeat run).
  kld.py dump LABEL [out_dir]    -> out_dir/LABEL.json.gz  (needs the server on :8001)
  kld.py cmp A.json.gz B.json.gz -> per-corpus ppl A/B, KLD(A||B) mean/p50/p99 (top-20 approx), top-1 agreement
Corpora: wikitext-2 test (OOD prose, 6 x 2048-token windows) and this repo's own code (written after the model's cutoff,
3 x 2048). KLD over A's top-20: tokens missing from B's top-20 get B's 20th logprob (upper-bounds B's mass there)."""
import gzip,json,os,re,sys,glob,urllib.request
try:import numpy as np
except ImportError:os.execv('/data/Jarrel/nqenv/bin/python',['/data/Jarrel/nqenv/bin/python',os.path.abspath(__file__)]+sys.argv[1:])
D=os.path.dirname(os.path.abspath(__file__));W=2048
def api(path,body):
    K=re.search(r'VLLM_API_KEY=(\S+)',open('/home/jarrelscy/homeassistant/.env').read()).group(1)
    r=urllib.request.Request('http://localhost:8001'+path,headers={'Authorization':'Bearer '+K,'Content-Type':'application/json'},data=json.dumps(body).encode())
    return json.load(urllib.request.urlopen(r,timeout=1800))
def corpora():
    wt=open(D+'/kld/wikitext2_test.txt').read()
    code=''.join(open(f).read() for f in sorted(glob.glob('/data/Jarrel/nestquant/sm120/*.py')+glob.glob('/data/Jarrel/nestquant/streaming/*.py')))
    return {'wikitext':(wt,6),'code':(code,3)}
def dump(lab,out):
    res={}
    for name,(txt,n) in corpora().items():
        ids=api('/tokenize',dict(model='local',prompt=txt[:n*W*8],add_special_tokens=False))['tokens']
        assert len(ids)>=n*W,(name,len(ids))
        ws=[]
        for i in range(n):
            p=ids[i*W:(i+1)*W]
            r=api('/v1/completions',dict(model='local',prompt=p,max_tokens=1,temperature=0,prompt_logprobs=20))
            pl=r['choices'][0]['prompt_logprobs'][1:]
            ws.append([[d[str(t)]['logprob'],{k:v['logprob'] for k,v in d.items()}] for t,d in zip(p[1:],pl)])
            print(name,i,'nll %.4f'%(-np.mean([x[0] for x in ws[-1]])),flush=True)
        res[name]=ws
    os.makedirs(out,exist_ok=True);json.dump(res,gzip.open(f'{out}/{lab}.json.gz','wt'))
def cmp(a,b):
    A=json.load(gzip.open(a,'rt'));B=json.load(gzip.open(b,'rt'));o={}
    for c in A:
        k=[];t1=[];na=[];nb=[]
        for wa,wb in zip(A[c],B[c]):
            for (la,da),(lb,db) in zip(wa,wb):
                na.append(-la);nb.append(-lb)
                ka=sorted(da,key=da.get)[-20:];fl=min(db.values())
                pa=np.exp([da[x] for x in ka]);pa/=pa.sum()
                la_=np.log(pa);lb_=np.array([db.get(x,fl) for x in ka]);lb_-=np.log(np.exp(lb_).sum())
                k.append(float((pa*(la_-lb_)).sum()));t1.append(max(da,key=da.get)==max(db,key=db.get))
        k=np.array(k);o[c]=dict(ntok=len(k),ppl_a=round(float(np.exp(np.mean(na))),4),ppl_b=round(float(np.exp(np.mean(nb))),4),
            kld=round(float(k.mean()),5),kld_se=round(float(k.std()/np.sqrt(len(k))),5),kld_p50=round(float(np.median(k)),5),
            kld_p99=round(float(np.quantile(k,.99)),4),top1=round(float(np.mean(t1)),4))
    print(json.dumps(dict(a=os.path.basename(a),b=os.path.basename(b),**o)))
if __name__=='__main__':
    if sys.argv[1]=='dump':dump(sys.argv[2],sys.argv[3] if len(sys.argv)>3 else D+'/kld')
    else:cmp(sys.argv[2],sys.argv[3])
