"""CPU test of nq_session (per-session floating-set restore): session key on tb4 prompts, snapshot/restore exactness,
diff-only + layer-ordered restore ops, pin vs scheduler/lookahead downs, unknown-session fallback, and the served level-4
share of the first 64 decode tokens after a switch (off vs on) on a 2-session alternating replay built from real decode
routing (session B = session A's routing with the non-fixed expert ids permuted per layer).
run: PYTHONPATH=/data/Jarrel/nq-dev/pylgb /data/Jarrel/nqenv/bin/python sm120/serve/test_session_cpu.py"""
import os,sys,types,json,glob,logging
import numpy as np
logging.basicConfig(level=logging.WARNING)
HERE=os.path.dirname(os.path.abspath(__file__));R=os.path.dirname(os.path.dirname(HERE))
sys.path[:0]=[HERE,R+'/streaming',R+'/sm120']
os.environ.setdefault('NQ_SR_OUT','/tmp/sr_test.jsonl');os.environ.setdefault('NQ_SR_CTL','/tmp/sr_test_ctl')
os.environ.update(NQ_GBDT_MODE=os.environ.get('NQ_GBDT_MODE','sync'),NQ_GBDT_SCALE='none',NQ_GBDT_THREADS='2')
import nq_session as SR,scheduler as SC,fixed_set as FS
ROUT='/data/Jarrel/nq-eval/results/c2-gbdt-cap24_routing';NE=256;TOPK=8

def key_test():
    from transformers import AutoTokenizer
    T=AutoTokenizer.from_pretrained('/data/models/jarrelscy/GLM-5.3-Vision-NVFP4-ARVQ-hybrid-pv-d4a105dc50dd')
    ks={};n=0
    for tj in sorted(glob.glob('/home/jarrelscy/homeassistant/benchmarks/glm5.3-arvq-v2-tb40-8h-c1-20260923/*/agent/trajectory.json'))[:14]:
        st=json.load(open(tj))['steps'];m=[s for s in st if s.get('source')=='user' or 'message' in s]
        u1=st[0]['message'];u1=u1 if isinstance(u1,str) else json.dumps(u1)
        a1=next((s.get('message') for s in st[1:] if s.get('source')=='agent'),'ok');a1=a1 if isinstance(a1,str) else json.dumps(a1)
        conv=[dict(role='user',content=u1)]
        t1=T(T.apply_chat_template(conv,tokenize=False,add_generation_prompt=True),add_special_tokens=False)['input_ids']
        t2=T(T.apply_chat_template(conv+[dict(role='assistant',content=a1),dict(role='user',content='New Terminal Output:\n$ ls\n')],
                                    tokenize=False,add_generation_prompt=True),add_special_tokens=False)['input_ids']
        k1,l1=SR.session_key(t1);k2,l2=SR.session_key(t2)
        assert k1==k2 and l1==l2 and l1<len(t1),(tj,l1,l2,len(t1))
        ks[k1]=tj;n+=1
    assert len(ks)==n,'tasks collide'
    print(f'key: {n} tasks, distinct keys, turn-1 key == turn-2 key; key lengths ok')

def routing(NL=12,T=3000):
    Ls=sorted(int(f[1:-4]) for f in os.listdir(ROUT))[:NL]
    g=np.stack([np.load(f'{ROUT}/L{L}.npy')[:T].astype(np.int64) for L in Ls])     # [NL, T, 8]
    return Ls,g

class FakeX:
    """lands every up / releases every down at the next poll (tracks the slot pool like RankExecutor)"""
    def __init__(s,nslot,rb=1<<20):s.rb=rb;s.q=[];s.nslot=nslot;s.calls=[]
    def apply(s,ups,downs,S):s.calls.append((list(ups),list(downs)));s.q+= [('u',x) for x in ups]+[('d',x) for x in downs]
    def poll(s,S):
        for k,(L,e) in s.q:(S.landed if k=='u' else S.released)(L,e)
        s.q=[]

def run(on,Ls,gA,gB,fx,nturn=4,dec=400,pf=600,check=False):
    open(os.environ['NQ_SR_CTL'],'w').write(f'on={int(on)}\n')
    dflt={L:[e for e in range(NE) if e not in fx[L]][:51] for L in Ls}
    S=SC.Scheduler(Ls,fx,dflt,1<<20,NE=NE,n_float=51,slots=56*len(Ls),cap_GBps=1e6,predictor='gbdt')
    for L in Ls:
        for e in dflt[L]:S.state[S.li[L],e]=2
    X=FakeX(56*len(Ls));sr=SR.SessionRestore(None);sr._ctl()
    pos={'A':0,'B':0};shares=[];rid=0;saved={}
    prompts={'A':[1,2,3,154828,9],'B':[1,2,4,154828,9]}
    def so(new=(),ns=None):return types.SimpleNamespace(scheduled_new_reqs=list(new),num_scheduled_tokens=ns or {})
    for turn in range(nturn):
        for sname,g in (('A',gA),('B',gB)):
            rid+=1;r=str(rid);ids=prompts[sname]+[7]*pf;p=pos[sname]
            sr.on_sched(so([types.SimpleNamespace(req_id=r,prompt_token_ids=ids,num_computed_tokens=0)],{r:len(ids)}))
            sr.service(S,X,None);X.poll(S)
            if check and on and turn>=1 and sname=='A':
                A=sr.A;assert A['switch'] and A['known'],A
                ups,downs=X.calls[-1] if X.calls else ([],[])
                li=[S.li[L] for L,_ in ups];assert li==sorted(li),'restore ups not layer ordered'
                T=A['tgt']['set']
                assert all(T[S.li[L],e] for L,e in ups) and not any(T[S.li[L],e] for L,e in downs),'restore ops not the diff'
                # prefill residue: a big step with the pin active must not down pinned experts
                c=np.zeros((len(Ls),NE));c[:, :200]=40
                S.step(c,600);assert not ((S.state==3)&T).any(),'pinned expert downed'
            # prefill counts (big step), then decode
            c=np.zeros((len(Ls),NE));np.add.at(c,(np.arange(len(Ls))[:,None],g[:,p:p+pf].reshape(len(Ls),-1)),1)
            ntok=sr.on_counts(S,c,pf,S.fixed|(S.state==2));S.step(c,ntok)
            sr.on_sched(so((),{r:1}))       # first decode step scheduled -> 'decode'
            sr.service(S,X,None)
            if check and on and turn>=1 and sname=='A':
                pd=sr.A['tgt']['P']
            w4=wa=0
            for t in range(dec):
                X.poll(S);sr.service(S,X,None)
                row=g[:,p+pf+t];c=np.zeros((len(Ls),NE));np.add.at(c,(np.arange(len(Ls))[:,None],row),1)
                lv=S.fixed|(S.state==2)
                if t==0 and check and on and turn>=1 and sname=='A':
                    ntok=sr.on_counts(S,c,1,lv)
                    for k in ('E','Et','nblk','S'):
                        a,b=getattr(S.P,k),pd[k]
                        assert (np.array_equal(a,b) if isinstance(a,np.ndarray) or isinstance(b,np.ndarray) else (a==b) if not isinstance(a,list) else all(np.array_equal(x,y) for x,y in zip(a,b))),k
                else:ntok=sr.on_counts(S,c,1,lv)
                if t<64:w4+=c[lv].sum();wa+=c.sum()
                ups,downs=S.step(c,ntok);X.apply(ups,downs,S)
            pos[sname]+=pf+dec
            if turn>=1:shares.append((sname,w4/wa))
    X.poll(S);sr.service(S,X,None)
    return shares,sr

def main():
    try:key_test()
    except AssertionError:raise
    except Exception as e:print('key test skipped:',repr(e)[:200])
    Ls,g=routing();fx,_,_=FS.load(layers=Ls)
    rng=np.random.default_rng(0);gB=g.copy()
    for i,L in enumerate(Ls):
        nf=np.array([e for e in range(NE) if e not in fx[L]]);perm=np.arange(NE);perm[nf]=rng.permutation(nf);gB[i]=perm[g[i]]
    half=g.shape[1]//2;gA,gB=g[:,:half],gB[:,half:]      # different content too
    for f in (os.environ['NQ_SR_OUT'],):
        if os.path.exists(f):os.remove(f)
    off,_=run(False,Ls,gA,gB,fx,nturn=2,dec=300,pf=300)
    on,sr=run(True,Ls,gA,gB,fx,nturn=2,dec=300,pf=300,check=True)
    rows=[json.loads(l) for l in open(os.environ['NQ_SR_OUT'])]
    so=[r for r in rows if r['on'] and r['switch'] and r['known']];assert so,'no known switch logged'
    for r in so:assert r['ups']>0 and r['win_share4'] is not None,r
    mo=np.mean([x for _,x in off]);mn=np.mean([x for _,x in on])
    print(f'first-64 decode level-4 share after a switch: off {mo:.4f} on {mn:.4f} ({len(on)} switches); stats {dict(sr.stats)}')
    print('sample row:',json.dumps(so[-1]))
    assert mn>mo+0.02,(mo,mn)
    # lookahead honours the pin
    import nq_lookahead as LAH
    S=SC.Scheduler(Ls,fx,{L:[e for e in range(NE) if e not in fx[L]][:51] for L in Ls},1<<20,NE=NE,slots=56*len(Ls),cap_GBps=1e6,predictor='ema')
    i=0;L=Ls[0];S.state[i]=0;fl=[e for e in range(NE) if e not in fx[L]];S.state[i,fl[:56]]=2
    S.pin=np.zeros_like(S.fixed);S.pin[i,fl[:40]]=True
    b=np.zeros(NE,np.float32);b[fl[60:]]=np.arange(len(fl[60:]),0,-1)
    fake=types.SimpleNamespace(rt=types.SimpleNamespace(SR=types.SimpleNamespace(restoring=lambda:True)))
    ups,downs,dfr,ch=LAH.LA._plan(fake,S,L,b,45)
    assert not any(S.pin[i,e] for _,e in downs) and len(ups)<=8,(len(ups),downs)
    print(f'lookahead with pin: {len(ups)} ups (cap 8 while restoring), {len(downs)} downs, none pinned')
    print('PASS')

if __name__=='__main__':main()
