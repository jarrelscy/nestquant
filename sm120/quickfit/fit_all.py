"""Quick weights-only NestQuant fit of every GLM-5.3 routed-expert layer (3..77; MTP layer 78 stays FP8).
Worker w of n takes layers L with (L - 3) % n == w. Per layer, per TP8 shard s (256 intermediate channels):
  BASE/L{L:02d}/tp{s}.pt   {E: {proj: {base, suh2, svh2}}}          2-bit base (resident, loaded once)
  RES/L{L:02d}/tp{s}.pt    {E: {proj: {p4, word, suh4, svh4}}}      P4 residual (streamed from NVMe)
  RES/L{L:02d}/manifest.json  proj_meta + fit settings. A layer is done when its manifest exists."""
import os,sys,json,time,torch
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import fit_quick as F
NE=F.NE;NSH=8
BASE=os.environ.get('NQ_BASE_OUT','/rawdata/Jarrel/nq-glm53-quick/base')
RES=os.environ.get('NQ_RES_OUT','/data/Jarrel/nq-glm53-quick/res')
w,n=int(sys.argv[1]),int(sys.argv[2])
layers=[L for L in range(3,78) if (L-3)%n==w]
def split(art):
    b=[{} for _ in range(NSH)];r=[{} for _ in range(NSH)]
    for pn in NE.PROJ:
        P=art[pn];m=P['meta'];k,nn=m['k'],m['n']
        for s in range(NSH):
            ks,ns=(slice(s*k//NSH,(s+1)*k//NSH),slice(0,nn)) if pn=='down' else (slice(0,k),slice(s*nn//NSH,(s+1)*nn//NSH))
            b[s][pn]=dict(base=P['base']['shards'][s],suh2=P['base']['suh'][ks].clone(),svh2=P['base']['svh'][ns].clone())
            r[s][pn]=dict(p4=P['p4']['shards'][s],word=P['p4']['word'][s].to(torch.int32),suh4=P['p4']['suh'][ks].clone(),svh4=P['p4']['svh'][ns].clone())
    return b,r
for L in layers:
    if os.path.exists(f'{RES}/L{L:02d}/manifest.json'):print('skip',L,flush=True);continue
    t=time.time();B=[{} for _ in range(NSH)];Rr=[{} for _ in range(NSH)];pm=None;info={}
    for E in range(int(os.environ.get('NQ_NEXP','256'))):
        art,_=F.fit(L,E)
        b,r=split(art)
        for s in range(NSH):B[s][E]=b[s];Rr[s][E]=r[s]
        pm=pm or {pn:{k:v for k,v in art[pn]['meta'].items()} for pn in NE.PROJ}
        info[E]={pn:art['meta']['info'][pn].get('bits') for pn in NE.PROJ}
        if E%64==63:print(f'L{L} E{E} {time.time()-t:.0f}s',flush=True)
    for root,parts in ((BASE,B),(RES,Rr)):
        os.makedirs(f'{root}/L{L:02d}',exist_ok=True)
        for s in range(NSH):torch.save(parts[s],f'{root}/L{L:02d}/tp{s}.pt.tmp');os.replace(f'{root}/L{L:02d}/tp{s}.pt.tmp',f'{root}/L{L:02d}/tp{s}.pt')
    json.dump(dict(layer=L,source=F.SRC,fit='weights-only quick (H=I, G=None, inner=0, base_var=None, single pass)',
                   res_K=F.RES_K,rate=art['meta']['rate'],proj_meta=pm,bits=info[0],sec=round(time.time()-t,1)),
              open(f'{RES}/L{L:02d}/manifest.json','w'),indent=1,default=str)
    print('done',L,round(time.time()-t),'s',flush=True)
