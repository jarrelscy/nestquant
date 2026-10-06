"""KLD of GLM-5.3 NestQuant builds vs the malaiwah BF16 reference, registry panel panel--glm53.malaiwah.corpus5x5-v1.

Reference: panel capture hidden [2047,6144] bf16 (post final norm) @ panel head (bf16) in fp32, log_softmax float64.
Candidate: layer-streamed FP8 backbone (eval_fp8 protocol: FP8->bf16, bf16 math, indexer skipped, SEQ 2048 = index_topk),
each context one causal 2048 sequence from position 0 (no BOS / template), FP8 model.norm + lm_head, positions 0..2046.
KL(ref||cand) per position over the full vocab in float64, no top-k.  Attention: fp8_ds_mla KV emulated (kvq.py, 'kvq')
on every stream except fp8_nokv.  Only routed experts differ between streams.

Streams (name -> source/levels):
  fp8 / fp8_nokv          FP8 experts (kvq / bf16 HF attention)
  <b>_all4                every expert level 4
  <b>_jf<H>[k0|fx]        jF predictor (GPUJointPredictor, jF.pt + v2 trees, hm 0.7), cold start, 16-token swaps, causal:
                          block b served by the set chosen after blocks < b (instant landing); fx = thread-22 fixed 26 +
                          (H-26) floating, start = serve floating_default; k0 = no fixed set, start = (fixed26+fd51)[:H]
  <b>_orc<H>              oracle: block's own salience top-H (all experts) at level 4
  <b>_prev<H>             previous block's salience top-H; block 0 = cold start of the build's serve arm
  b = b24 (2/4 repack) | b175 (1.75/4 repack).  salience = sum w^2 |x|^2 (w incl. routed_scaling_factor, x post-LN MoE in)
  torchrun --nproc_per_node 4 eval_reg.py --streams fp8,fp8_nokv --tag val [--n-layers N] [--contexts 0,1,2,3]"""
import argparse,json,math,os,re,sys,time
import numpy as np,torch,torch.distributed as dist
HERE=os.path.dirname(os.path.abspath(__file__));REPO=os.path.dirname(os.path.dirname(HERE))
for p in (HERE,REPO+'/sm120',REPO+'/streaming',REPO+'/threads/34-tr3','/data/Jarrel/nq-serve/predictor/joint','/data/Jarrel/nq-kld/reg/lgbstub'):
    if p not in sys.path:sys.path.insert(0,p)
import eval_fp8 as EF,nq_io,nqeff,fixed_set as FS,kvq
from safetensors.torch import load_file
PANEL='/data/Jarrel/nq-kld/reg/root';PRIV='/data/Jarrel/nq-kld/reg/private'
REPACK={'b24':'/home/jarrelscy/nq-p4rec/hf','b175':'/home/jarrelscy/nq-175/hf'}
JD='/data/Jarrel/nq-serve/predictor/joint';HM=0.7
CARD={'b24':(2.025757,4.141464),'b175':(1.775757,4.141464)}   # model-card accounting (threads/35-nq15/results/sizing.json): base incl. scales / level 4 incl. residual + low-rank
NE,TOPK,SEQ,G=256,8,2048,16;NB=SEQ//G
RANK=int(os.environ.get('RANK','0'));WORLD=int(os.environ.get('WORLD_SIZE','1'))
def log(m):
    if RANK==0:print(f'[r{RANK} {time.strftime("%H:%M:%S")}] {m}',flush=True)
def parse(s):
    if s in ('fp8','fp8_nokv'):return dict(src=None,kind='fp8')
    m=re.fullmatch(r'(b24|b175)_(all4|jf(\d+)(k0|fx)|orc(\d+)|prev(\d+))',s);assert m,s
    b=m.group(1)
    if m.group(2)=='all4':return dict(src=b,kind='all4',H=256)
    if m.group(3):return dict(src=b,kind='jf',H=int(m.group(3)),mode=m.group(4))
    H=int(m.group(5) or m.group(6));return dict(src=b,kind='orc' if m.group(5) else 'prev',H=H)

def block_stats(ids,sv,Nr):
    """ids [Nr*SEQ,8] long, sv [Nr*SEQ,8] f64 -> per-block counts f32 / salience f64 [Nr,NB,NE]"""
    i=ids.view(Nr,NB,G*TOPK);c=torch.zeros(Nr,NB,NE,dtype=torch.float32,device=ids.device).scatter_add_(2,i,torch.ones_like(i,dtype=torch.float32))
    s=torch.zeros(Nr,NB,NE,dtype=torch.float64,device=ids.device).scatter_add_(2,i,sv.view(Nr,NB,G*TOPK));return c,s
def topH(s,H):
    o=torch.argsort(-s,dim=-1,stable=True)[...,:H];m=torch.zeros_like(s,dtype=torch.bool);m.scatter_(-1,o,True);return m

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--streams',required=True);ap.add_argument('--tag',required=True)
    ap.add_argument('--n-layers',type=int,default=0);ap.add_argument('--contexts',default='')
    ap.add_argument('--attn-chunk',type=int,default=1);ap.add_argument('--pos-chunk',type=int,default=256)
    ap.add_argument('--ebatch',type=int,default=32);ap.add_argument('--dbatch',type=int,default=8)
    ap.add_argument('--out',default='/tmp/kldreg_results.json');a=ap.parse_args()
    dist.init_process_group('nccl');torch.cuda.set_device(int(os.environ.get('LOCAL_RANK',RANK)));dev=torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    pj=json.load(open(PANEL+'/panel/panel.json'));dom={r['context_index']:r['domain'] for r in pj['records']}
    ctx=[int(x) for x in a.contexts.split(',')] if a.contexts else list(range(len(dom)))
    NW=-(-len(ctx)//WORLD)*WORLD;wins=ctx+ctx[:NW-len(ctx)];real=[True]*len(ctx)+[False]*(NW-len(ctx))   # padding windows = duplicates, unscored
    own=list(range(RANK,NW,WORLD));Nr=len(own);Tr=Nr*SEQ;T=Tr*WORLD;myE=[e for e in range(NE) if e%WORLD==RANK]
    toks=np.stack([np.asarray(json.load(open(f'{PANEL}/panel/tokens/context-{c:04d}.json')),np.int64) for c in wins]);assert toks.shape[1]==SEQ
    for c in set(wins):assert np.load(f'{PANEL}/panel/masks/context-{c:04d}.npy').all()
    streams=[s for s in a.streams.split(',') if s];SP={s:parse(s) for s in streams};S_=len(streams)
    cfg=EF.load_config();nl=a.n_layers or cfg.num_hidden_layers;fp8=nq_io.FP8Model(EF.FP8);bb=EF.Backbone(cfg,fp8,dev)
    srcs=sorted({p['src'] for p in SP.values() if p['src']})
    nql={b:sorted(int(x) for x in json.load(open(f'{REPACK[b]}/rank0.json'))['layers']) for b in REPACK}
    fx26,fsrc,_=FS.load(layers=nql['b24']);fj=json.load(open(FS.T22))
    def nr_top(L,excl,n):return [int(x) for x in np.argsort(-np.where(np.isin(np.arange(NE),excl),-1,np.array(fj['n_routed'][str(L)])))[:n]]
    def k0_start(L,H):return ([int(e) for e in fj['fixed_set'][str(L)]]+[int(e) for e in fj['floating_default'][str(L)]])[:H]
    def plan(s,L):     # -> fixed list, start floating list, n_float
        p=SP[s]
        if p['kind']=='jf' and p['mode']=='fx':fx=[int(e) for e in fx26[L]];return fx,nr_top(L,fx,p['H']-len(fx)),p['H']-len(fx)   # == nq_vllm NQ_JOINT_FIXED=1 dflt
        if p['kind'] in ('jf','prev'):
            if p['kind']=='prev' and p['src']=='b24':fx=[int(e) for e in fx26[L]];return [],fx+nr_top(L,fx,p['H']-len(fx)),p['H']   # 2/4 released cold start
            return [],k0_start(L,p['H']),p['H']
        return [],[],0
    log(f'contexts {len(ctx)} -> {NW} windows ({Nr}/rank); streams {streams}; layers {nl}; tag {a.tag}')
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaRotaryEmbedding,GlmMoeDsaRMSNorm
    import gpu_predictor as GJ
    rot=GlmMoeDsaRotaryEmbedding(cfg).to(dev);pos=torch.arange(SEQ,device=dev).view(1,-1)
    cos_sin=rot(torch.empty(1,SEQ,1,device=dev,dtype=torch.bfloat16),pos)
    topk_all=pos.to(torch.int32).view(1,1,SEQ).expand(a.attn_chunk,SEQ,SEQ)
    ids_own=torch.from_numpy(toks[own]).to(dev)
    emb=fp8.tensor('model.embed_tokens.weight',dev);h0=torch.nn.functional.embedding(ids_own,emb).to(torch.bfloat16);del emb
    hid=[h0]+[h0.clone() for _ in streams[1:]];torch.cuda.empty_cache()
    realtok=torch.tensor([real[w] for w in own],device=dev).view(Nr,1).expand(Nr,SEQ).clone();realtok[:,SEQ-1]=False;realtok=realtok.reshape(-1)
    SH={s:torch.zeros(4,dtype=torch.float64,device=dev) for s in streams if SP[s]['src']}   # cnt4, cnt, sal4, sal (own real tokens, NQ layers)
    tl={};t_start=time.time();tpred=0.
    for li in range(nl):
        t0=time.time();layer,sparse=bb.build(li);attn=layer.self_attn;kvq.install(attn,'kvq')
        with torch.no_grad():
            for si,h in enumerate(hid):
                fwd=(lambda **k:type(attn).forward(attn,**k)) if streams[si]=='fp8_nokv' else attn.forward
                for s0 in range(0,Nr,a.attn_chunk):
                    s1=min(s0+a.attn_chunk,Nr)
                    att=fwd(hidden_states=layer.input_layernorm(h[s0:s1]),position_embeddings=cos_sin,attention_mask=None,
                            position_ids=pos.expand(s1-s0,-1),prev_topk_indices=topk_all[:s1-s0])[0]
                    h[s0:s1]+=att;del att
            flats=[h.view(Tr,-1) for h in hid]
            if not sparse:
                for f in flats:
                    for c0 in range(0,Tr,8192):f[c0:c0+8192]+=layer.mlp(layer.post_attention_layernorm(f[c0:c0+8192]))
            else:
                t_moe=time.time();xs=[layer.post_attention_layernorm(f) for f in flats];R=[layer.mlp.gate(x) for x in xs]
                ids_l=[r[2].long() for r in R];w_l=[r[1].float() for r in R];del R
                nq={s:(SP[s]['src'] is not None and li in nql[SP[s]['src']]) for s in streams}
                lvl={}
                for si,s in enumerate(streams):   # own-window level-4 masks [Tr,8]
                    if not nq[s]:continue
                    p=SP[s];tp=time.time()
                    if p['kind']=='all4':m4=torch.ones(Tr,TOPK,dtype=torch.bool,device=dev);sv=None
                    else:
                        xn=torch.cat([xs[si][c0:c0+8192].float().pow(2).sum(-1) for c0 in range(0,Tr,8192)]).double()
                        sv=w_l[si].double().pow(2)*xn[:,None];del xn
                        cnt,sal=block_stats(ids_l[si],sv,Nr)
                        if p['kind']=='orc':setb=topH(sal,p['H'])
                        elif p['kind']=='prev':
                            _,st,_=plan(s,li);s0_=torch.zeros(Nr,1,NE,dtype=torch.bool,device=dev);s0_[:,:,st]=True
                            setb=torch.cat([s0_,topH(sal[:,:-1],p['H'])],1)
                        else:
                            fx,st,nf=plan(s,li);P=GJ.GPUJointPredictor([li]*Nr,{li:fx},JD+'/jF.pt',n_float=nf,hm=HM,device=dev,v2_model=JD+'/v2_sal_tweedie1.5.txt',graph=False)
                            fxm=np.zeros((Nr,NE),bool);fxm[:,fx]=True;want=np.zeros((Nr,NE),bool);want[:,st]=True;want&=~fxm
                            sb=np.zeros((Nr,NB,NE),bool)
                            for b in range(NB):
                                sb[:,b]=want|fxm
                                if P.step(cnt[:,b],G,None,b==0,sal=sal[:,b]):
                                    w=P.target(want)
                                    if w is not None:want=w&~fxm
                            setb=torch.from_numpy(sb).to(dev);del P
                        bidx=(torch.arange(Tr,device=dev)//SEQ*NB+torch.arange(Tr,device=dev)%SEQ//G)
                        m4=setb.reshape(Nr*NB,NE)[bidx].gather(1,ids_l[si]);del setb,cnt,sal
                    tpred+=time.time()-tp
                    mr=m4[realtok].double();SH[s][0]+=mr.sum();SH[s][1]+=mr.numel()
                    if sv is not None:svr=sv[realtok];SH[s][2]+=(mr*svr).sum();SH[s][3]+=svr.sum();del sv,svr
                    else:SH[s][2]+=1;SH[s][3]+=1
                    M=torch.empty(T,TOPK,dtype=torch.uint8,device=dev);dist.all_gather_into_tensor(M,m4.to(torch.uint8).contiguous());lvl[s]=M.bool();del m4
                gx=[];gi=[];gw=[]
                for si in range(S_):
                    X=torch.empty(T,xs[si].shape[1],dtype=xs[si].dtype,device=dev);dist.all_gather_into_tensor(X,xs[si].contiguous());gx.append(X)
                    I_=torch.empty(T,TOPK,dtype=torch.long,device=dev);dist.all_gather_into_tensor(I_,ids_l[si].contiguous());gi.append(I_)
                    W_=torch.empty(T,TOPK,dtype=torch.float32,device=dev);dist.all_gather_into_tensor(W_,w_l[si].contiguous());gw.append(W_)
                del ids_l,w_l
                ys=[torch.zeros(T,xs[0].shape[1],dtype=torch.float32,device=dev) for _ in range(S_)]
                order=[];offs=[]
                for si in range(S_):
                    fl=gi[si].reshape(-1);order.append(torch.argsort(fl,stable=True));offs.append([0]+torch.bincount(fl,minlength=NE).cumsum(0).tolist())
                need={}
                for s in streams:
                    if nq[s]:need.setdefault(SP[s]['src'],set()).update({4} if SP[s]['kind']=='all4' else {2,4})
                t_dec=0.
                for b0 in range(0,len(myE),a.ebatch):
                    Eb=myE[b0:b0+a.ebatch];td=time.time();NQW={}
                    for src,lvs in need.items():
                        parts={(E,lv):[] for E in Eb for lv in lvs}
                        for k in range(4):
                            RL=nqeff.RankLayerEff(REPACK[src],li,k,dev,hash_=False)
                            for lv in lvs:
                                for E,(Gm,U,D) in RL.experts(Eb,lv,a.dbatch).items():parts[(E,lv)].append((Gm.bfloat16(),U.bfloat16(),D.bfloat16()))
                            del RL
                        for (E,lv),p_ in parts.items():NQW[(src,E,lv)]=(torch.cat([x[0] for x in p_]),torch.cat([x[1] for x in p_]),torch.cat([x[2] for x in p_],1))
                        del parts
                    t_dec+=time.time()-td
                    for E in Eb:
                        Wr=fp8.expert(li,E,dev);Wr=(Wr['gate_proj'],Wr['up_proj'],Wr['down_proj'])
                        for si,s in enumerate(streams):
                            a_,b_=offs[si][E],offs[si][E+1]
                            if a_==b_:continue
                            sl=order[si][a_:b_];tk=sl//TOPK;wt=gw[si].reshape(-1)[sl]
                            if not nq[s]:groups=[(tk,wt,Wr)]
                            else:
                                src=SP[s]['src'];m4=lvl[s].reshape(-1)[sl];groups=[(tk[m4],wt[m4],NQW[(src,E,4)])]
                                if (src,E,2) in NQW:groups.append((tk[~m4],wt[~m4],NQW[(src,E,2)]))
                                else:assert bool(m4.all())
                            for tk_,wt_,W in groups:
                                if len(tk_):ys[si].index_add_(0,tk_,EF.ffn(gx[si][tk_],*W).float()*wt_[:,None])
                        del Wr
                    NQW.clear()
                del gx,gi,gw,order,lvl;torch.cuda.empty_cache()
                for si in range(S_):
                    out=torch.empty(Tr,ys[si].shape[1],dtype=torch.float32,device=dev);dist.reduce_scatter_tensor(out,ys[si]);ys[si]=None
                    flats[si]+=(out+layer.mlp.shared_experts(xs[si]).float()).to(flats[si].dtype);del out
                del xs,ys;log(f'layer {li}: moe {time.time()-t_moe:.1f}s (decode {t_dec:.1f}s, pred cum {tpred:.0f}s)')
        del layer,attn;torch.cuda.empty_cache();tl[li]=time.time()-t0
        log(f'layer {li} {tl[li]:.1f}s peak {torch.cuda.max_memory_allocated()/2**30:.1f}G elapsed {(time.time()-t_start)/60:.1f}m')
    # ---------------------------------------------------------------- scoring vs BF16 reference (own real windows)
    norm=GlmMoeDsaRMSNorm(cfg.hidden_size,cfg.rms_norm_eps).to(dev);norm.load_state_dict({'weight':fp8.tensor('model.norm.weight',dev)})
    head=load_file(PANEL+'/head/weight.safetensors')['lm_head.weight'].to(dev).float()
    lg=lambda x:x.float()@head.T
    mine={}
    with torch.no_grad():
        for j,w in enumerate(own):
            if not real[w]:continue
            c=wins[w];hr=load_file(f'{PANEL}/capture/hidden_{c:04d}.safetensors')['hidden_states'].to(dev);assert hr.shape==(SEQ-1,cfg.hidden_size)
            hc=[norm(hid[si][j])[:-1] for si in range(S_)];kl=[[] for _ in hc];ag=[[] for _ in hc]
            for p0 in range(0,SEQ-1,a.pos_chunk):
                p1=min(p0+a.pos_chunk,SEQ-1);lr=torch.log_softmax(lg(hr[p0:p1]).double(),-1);am=lr.argmax(-1);pr=lr.exp()
                for si,h in enumerate(hc):
                    lc=torch.log_softmax(lg(h[p0:p1]).double(),-1);kl[si].append((pr*(lr-lc)).sum(-1));ag[si].append(lc.argmax(-1)==am);del lc
                del lr,pr
            mine[c]={s:(torch.cat(kl[si]).cpu().numpy(),torch.cat(ag[si]).cpu().numpy()) for si,s in enumerate(streams)}
    for s in SH:dist.all_reduce(SH[s])
    allr=[None]*WORLD;dist.all_gather_object(allr,mine)
    if RANK==0:
        per={};[per.update(r) for r in allr];cs=sorted(per);assert cs==sorted(ctx),(cs,ctx)
        os.makedirs(PRIV,exist_ok=True)
        np.savez_compressed(f'{PRIV}/{a.tag}_tok.npz',**{f'{s}_kl_{c:04d}':per[c][s][0] for c in cs for s in streams},**{f'{s}_top1_{c:04d}':per[c][s][1] for c in cs for s in streams})
        bits={b:EF.bits_table(argparse.Namespace(repack=REPACK[b],aqlm=None,arvq=None),nql[b]) for b in srcs}
        arms={}
        for s in streams:
            k=np.concatenate([per[c][s][0] for c in cs]);t1=np.concatenate([per[c][s][1] for c in cs]);p=SP[s]
            doms=sorted({dom[c] for c in cs})
            r=dict(build={'b24':'2/4','b175':'b1.75/4',None:'fp8'}[p['src']],kind=p['kind'],H=p.get('H'),n_positions=int(len(k)),kld=float(k.mean()),
                   kld_se_ctx=float(np.std([per[c][s][0].mean() for c in cs])/math.sqrt(len(cs))),top1=float(t1.mean()),
                   kld_p50=float(np.median(k)),kld_p99=float(np.quantile(k,.99)),
                   per_domain={d:dict(kld=float(np.concatenate([per[c][s][0] for c in cs if dom[c]==d]).mean()),top1=float(np.concatenate([per[c][s][1] for c in cs if dom[c]==d]).mean())) for d in doms},
                   per_context={str(c):dict(domain=dom[c],kld=float(per[c][s][0].mean()),top1=float(per[c][s][1].mean())) for c in cs},
                   kv='bf16 HF attention' if s=='fp8_nokv' else 'fp8_ds_mla emulated (kvq)')
            if p['src']:
                v=SH[s].tolist();r.update(share4_calls=v[0]/v[1],share4_salience=v[2]/v[3])
                b2,b4=bits[p['src']]['nq2'],bits[p['src']]['nq4'];c2,c4=CARD[p['src']];r.update(resident_bpw=c2+p['H']/NE*(c4-c2),bpw_base_card=c2,bpw_level4_card=c4,resident_bpw_repack_bytes=b2+p['H']/NE*(b4-b2),bpw_level2_repack_bytes=b2,bpw_level4_repack_bytes=b4)
            arms[s]=r;log(f"{s:14s} KLD {r['kld']:.5f} top1 {100*r['top1']:.2f}%"+(f" share4 calls {r['share4_calls']:.3f} sal {r['share4_salience']:.3f} bpw {r['resident_bpw']:.3f}" if p['src'] else ''))
        res=json.load(open(a.out)) if os.path.exists(a.out) else dict(panel=pj['panel_id'],arms={},passes={})
        res['arms'].update(arms)
        res['passes'][a.tag]=dict(streams=streams,contexts=cs,n_layers=nl,wall_s=time.time()-t_start,finished_melbourne=os.popen('TZ=Australia/Melbourne date "+%F %H:%M %Z"').read().strip(),
                                 fixed_set_source=fsrc,repacks={b:REPACK[b] for b in srcs},smoke=bool(a.n_layers or a.contexts))
        json.dump(res,open(a.out,'w'),indent=1);log(f'wrote {a.out}; wall {(time.time()-t_start)/60:.1f} min')
    dist.barrier();dist.destroy_process_group()

if __name__=='__main__':main()
