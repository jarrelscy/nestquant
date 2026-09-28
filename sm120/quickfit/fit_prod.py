"""Production-FORMAT dummy fit of GLM-5.3 routed experts (format/speed testing; accuracy is not the goal).
Same bytes layout as threads/12 nq_layer.py (format nestquant-v1): base_var 'sign', residual gate/up K2 + down K2.3125
(0x9248) = 4.1263 bpw, OUT/L{L}/tp{s}.pt {E: {proj: {base, var, p4, word, suh2, svh2, suh4, svh4}}} + manifest.json.
Differences from a real production fit (values only, not layout): H = identity, G = None, inner = 0, single pass
(canonical_base False), and the pattern-rate residual uses a greedy tail-biting trellis (greedy_patq) instead of the torch
Viterbi (~80 s/expert). The greedy ring is exactly self-consistent (last 16 state bits forced to the start state), so the
packed stream decodes to the encoder's states and nq_decode / the kernel read it as any other artifact."""
import os,sys,json,time,torch;torch.cuda.set_per_process_memory_fraction(12/96)
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import fit_quick as F
NE,PV,D=F.NE,F.PV,F.NE.D
import nq_layer as NL
NSH=8

@torch.no_grad()
def greedy_patq(tiles,K,chunk=None):
    """tiles [R,256] (Viterbi order) -> (values, states) like PV.patq; greedy per step, tail-biting consistent."""
    if PV._LUT is None:PV._LUT=F.NE.h.codebook_lut('mul1').float().cuda()
    V=PV._LUT;St=PV.vsteps(K);w=tiles.float();T=w.shape[0];dev=w.device
    def run(s,force=None):
        out=torch.empty(T,256,dtype=torch.int64,device=dev)
        for i in range(256):
            k=St[i];b=torch.arange(1<<k,device=dev)
            cand=((s[:,None]<<k)|b[None])&0xFFFF                          # [T, 2^k]
            c=(V[cand]-w[:,i,None]).square()
            if force is not None and i in force:
                o,m,tgt=force[i];c=c.masked_fill(((b[None]^tgt[:,None])&m)!=0,float('inf'))
            s=cand.gather(1,c.argmin(1,keepdim=True))[:,0];out[:,i]=s
        return out,s
    _,s0=run(torch.zeros(T,dtype=torch.int64,device=dev))
    force={};o=0
    for i in range(255,-1,-1):                                            # steps whose bits make up the final 16-bit state
        if o>=16:break
        force[i]=(o,(1<<min(St[i],16-o))-1,(s0>>o));o+=St[i]
    force={i:(o_,m,t&m) for i,(o_,m,t) in force.items()}
    st,s1=run(s0,force);assert torch.equal(s1,s0)
    return V[st],st
PV.patq=greedy_patq

RES_K={'gate':2.0,'up':2.0,'down':2.3125}
def fit(L,E):
    Ws=F.teacher(L,E);ey=lambda n:torch.eye(n,device='cuda')
    HG={'H':[ey(Ws[0].shape[1]),ey(Ws[0].shape[1]),ey(Ws[2].shape[1])],'G':[None,None,None]}
    art,dense=NE.encode_expert(Ws,HG,base_var='sign',inner=0,check=True,canonical_base=False,res_K=RES_K)
    art['meta'].update(layer=L,expert=E,flags=['dummy-fit: H=I G=none inner=0 single-pass greedy-pattern-residual'],bnd=None,hg_meta={})
    return art,dense,Ws

if __name__=='__main__' and sys.argv[1]=='one':
    torch.backends.cuda.matmul.allow_tf32=False
    L,E=int(sys.argv[2]),int(sys.argv[3]);t=time.time();art,dense,Ws=fit(L,E);dt=time.time()-t
    for i,pn in enumerate(NE.PROJ):
        print(pn,{lv:round(float((dense[lv][i]-Ws[i].cpu()).norm()/Ws[i].cpu().norm()),4) for lv in (2,4)},art['meta']['info'][pn]['bits'])
    print('rate',art['meta']['rate'],'sec',round(dt,1))
    torch.save(art,f'/data/Jarrel/nq-glm53-prod/smoke_L{L}_E{E}.pt')

def finalize(d,L,arts,extra=None):
    """{E: artifact} of layer L -> d/tp{s}.pt + d/manifest.json, byte-identical structure to nq_layer.py's finalize."""
    os.makedirs(d,exist_ok=True);have=sorted(arts)
    shards=[dict() for _ in range(NSH)];per_exp,proj_meta={},None
    for E in have:
        art=arts[E]
        for s,part in enumerate(NL.split_expert(art)):shards[s][E]=part
        m=art['meta']
        per_exp[E]=dict(flags=m.get('flags'),rate=m['rate'],bnd=m.get('bnd'),
                        proj={p:{k:m['info'][p][k] for k in ('bits','proxy_rot','bitexact','L2_equal_canonical') if k in m['info'][p]} for p in NE.PROJ})
        if proj_meta is None:
            proj_meta={p:{k:v for k,v in art[p]['meta'].items()} for p in NE.PROJ}
            cfg={k:m[k] for k in ('format','rate','base_var','lam','inner','sigma','canonical_base','res_K') if k in m};cfg['bnd']=m.get('bnd')
    files={}
    for s in range(NSH):
        f=f'{d}/tp{s}.pt';torch.save(shards[s],f+'.tmp');os.replace(f+'.tmp',f)
        files[f'tp{s}.pt']=dict(sha256=NL.sha(f),bytes=os.path.getsize(f))
    exp_bytes={}
    for pn in NE.PROJ:
        pr=shards[0][have[0]][pn]
        exp_bytes[pn]={k:int(v.numel()*v.element_size()) if k!='word' else 2*int(v.numel()) for k,v in pr.items()}
        if 'var' in pr:exp_bytes[pn]['var']=(pr['var'].numel()*D.variant_bits(cfg['base_var'])+7)//8
    man=dict(format='nestquant-v1',layer=L,n_experts=len(have),experts=have,config=cfg,tp=NSH,proj_meta=proj_meta,files=files,
             packed_bytes_per_expert_per_shard=exp_bytes,
             note="word = u16 Mb | N<<8 per 16x128 unit (stored int32 here); var = 1-bit per-ring sign "
                  "(stored uint8 here); base/p4 = LSB-first ring streams, units in (strip, chunk) order per shard; "
                  "residual K per unit from proj_meta.res_rule (positional, no map). Decoder: nq_decode.py.",
             per_expert=per_exp,time=time.strftime('%Y-%m-%d %H:%M:%S'),**(extra or {}))
    json.dump(man,open(f'{d}/manifest.json','w'),indent=1)
    return man

OUT=os.environ.get('NQ_PROD_OUT','/rawdata/Jarrel/nq-glm53-prod')
if __name__=='__main__' and sys.argv[1]=='all':
    torch.backends.cuda.matmul.allow_tf32=False
    w,n=int(sys.argv[2]),int(sys.argv[3]);nexp=int(os.environ.get('NQ_NEXP','256'))
    for L in [L for L in range(3,78) if (L-3)%n==w]:
        d=f'{OUT}/L{L}'
        if os.path.exists(f'{d}/manifest.json'):print('skip',L,flush=True);continue
        t=time.time();arts={}
        for E in range(nexp):
            arts[E],_,_=fit(L,E)
            if E%32==31:print(f'L{L} E{E} {time.time()-t:.0f}s',flush=True)
        finalize(d,L,arts,dict(dummy_fit='H=I G=none inner=0 single-pass, greedy pattern residual (format test, not accuracy)',source=F.SRC))
        print('done',L,round(time.time()-t),'s',flush=True);del arts
