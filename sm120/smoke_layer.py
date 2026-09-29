"""Whole-layer TP smoke on the production shard files: every rank of a TP=tp run loads its shard group of all experts of
layer L (nqload.RankLayer), levels = fixed set (threads/22) + random experts up to `hot` share at level 4, rest level 2,
random top-8 routing, B = 1..4. Sum over ranks of the kernel outputs vs the dense reference built independently from
all 8 shards (nq_layer.assemble -> nq_decode.decode_expert, lr included). Also reports load time and P4 slot size.
Layers with a threads/29 in_had_down (manifest config) are referenced through the lead's decoder nq29_had.decode_expert29;
env SMOKE_HAD=128 forces the kernel's down-input Hadamard to 128 (negative control: must FAIL on such a layer).
usage: smoke_layer.py ROOT L [tp=4] [hot=0.3]   (hot=1: every expert at level 4; hot=0: every expert at level 2)"""
import os,sys,json,time,random,torch;torch.cuda.set_per_process_memory_fraction(40/96)
HERE=os.path.dirname(os.path.abspath(__file__));sys.path[:0]=[HERE,HERE+'/../streaming']
import nqload as NQ;from moe import MoELayer
import nq_layer as NL,nq_decode as D
import fixed_set as FS
root,L=sys.argv[1],int(sys.argv[2]);tp=int(sys.argv[3]) if len(sys.argv)>3 else 4;hot=float(sys.argv[4]) if len(sys.argv)>4 else 0.3
dev='cuda';torch.backends.cuda.matmul.allow_tf32=False
t=time.time();ranks=[NQ.RankLayer(root,L,r,tp) for r in range(tp)];print(f'L{L} TP{tp}: loaded {len(ranks[0].experts)} experts x {tp} ranks in {time.time()-t:.1f}s',flush=True)
man=ranks[0].man;ids=man['experts'];NE_=max(ids)+1;H,I=ranks[0].H,ranks[0].I;HW=NQ.had_width(man)
if HW!=128:
    sys.path.append(HERE+'/../threads/29-outlier-gap');import nq29_had as NH
    print(f'  in_had_down {HW} (reference: nq29_had.decode_expert29)',flush=True)
if os.environ.get('SMOKE_HAD'):
    for r in ranks:
        for x in r.ex.values():x.had_dn=int(os.environ['SMOKE_HAD'])
    print(f'  NEGATIVE CONTROL: kernel in_had_down forced to {os.environ["SMOKE_HAD"]}',flush=True)
if os.path.exists(f'{root}/L{L}/tp0.pt'):ASM=NL.assemble
else:                                          # HF layout (thread-25 safetensors): the assembler wants ROOT = <repo>/layers
    sys.path.append(HERE+'/../threads/25-campaign');import nq25_st
    _ar=root if os.path.exists(f'{root}/L{L}/manifest.json') else f'{root}/layers'
    ASM=lambda _r,L_,E_:nq25_st.assemble(_ar,L_,E_)
fs=[e for e in FS.load(layers=[L])[0][L] if e in ids] if hot>0 else [];rng=random.Random(L)
l4=set(fs)|set(rng.sample([e for e in ids if e not in fs],max(0,min(len(ids)-len(fs),int(hot*len(ids))-len(fs)))))
levels=[4 if e in l4 else 2 for e in range(NE_)]
print(f'  level 4: {len(l4)} experts ({len(set(fs))} fixed); P4 slot per rank {[r.slot_bytes() for r in ranks]} B; '
      f'lr ranks gu/dn max {max(e.rg for e in ranks[0].ex.values())}/{max(e.rd for e in ranks[0].ex.values())}',flush=True)
layers=[]
for r in ranks:
    M=MoELayer(NE_,H,I,Bmax=4)
    for e,ex in r.ex.items():ex.signs=ex.sc[levels[e]];M.set(e,ex,levels[e])
    layers.append(M)
cache={}
def dense(E,lv):
    if (E,lv) not in cache:
        if HW==128:cache[E,lv]=[w.float() for w in D.decode_expert(ASM(root,L,E),lv,dev)]
        else:
            art=ASM(root,L,E);art['meta']=dict(art.get('meta') or {},in_had_down=HW)
            cache[E,lv]=[w.float() for w in NH.decode_expert29(art,lv,dev)]
    return cache[E,lv]
ok=True;worst=0;TK=min(8,len(ids))
for B in (1,2,3,4):
    for trial in range(2):
        cache.clear();torch.cuda.empty_cache()
        sel=torch.tensor([rng.sample(ids,TK) for _ in range(B)],device=dev)   # distinct experts per token, as the router
        rw=torch.softmax(torch.randn(B,TK,device=dev),1).half()
        x=(torch.randn(B,H,device=dev)*0.05).half()
        y=sum(M(x,sel,rw).float().clone() for M in layers)
        ref=torch.zeros(B,H,device=dev);xf=x.float()
        for b in range(B):
            for k in range(TK):
                E=int(sel[b,k]);Wg,Wu,Wd=dense(E,levels[E])
                ref[b]+=float(rw[b,k])*((torch.nn.functional.silu(xf[b]@Wg.T)*(xf[b]@Wu.T))@Wd.T)
        rel=((y-ref).norm()/ref.norm()).item();worst=max(worst,rel);ok&=rel<3e-3
        print(f'  B{B} trial {trial}: {sum(levels[int(e)]==4 for e in sel.flatten())}/{TK*B} picks at level 4, rel err {rel:.2e}',flush=True)
print(f'L{L} TP{tp} worst {worst:.2e}',('LAYER PASS' if ok else 'LAYER FAIL'))
