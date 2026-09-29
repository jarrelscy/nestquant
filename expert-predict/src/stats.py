import sys; sys.path.insert(0, '/data/Jarrel/expert-predict/src'); from common import *
for task in ['formal-crypto','freight-dispatch-shift']:
    d=load(task);tok=d['tok'];req=d['req']
    _,c=np.unique(req,return_counts=True)
    print(task,'decode reqs',len(c),'tok/req pct',np.percentile(c,[10,50,90]).astype(int),'mean',int(c.mean()))
    # segment state
    st=np.zeros(len(tok),np.int8);s=0;prev=-1
    for i,t in enumerate(tok):
        if req[i]!=prev: s=1; prev=req[i]   # decode starts in think by default? check
        if t==SPECIAL['think']:s=1
        elif t==SPECIAL['ethink']:s=2
        elif t==SPECIAL['tc']:s=3
        elif t==SPECIAL['etc']:s=2
        st[i]=s
    print(' state share think/answer/tool',[round((st==k).mean(),3) for k in (1,2,3)])
    print(' specials per 1k tok',{k:round(float((tok==v).sum())/len(tok)*1000,2) for k,v in SPECIAL.items()})
    # first 20 tokens of a request
    i0=np.nonzero(np.r_[True,req[1:]!=req[:-1]])[0][:3]
    for i in i0: print(' start tokens',tok[i:i+12])
