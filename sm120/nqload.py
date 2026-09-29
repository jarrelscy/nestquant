"""Loader for the production shard format (threads/12 nq_layer.py, nestquant-v1): L{L}/tp{s}.pt (TP8 shards) +
manifest.json -> sm120 kernel experts. Rank r of a TP=tp run owns the contiguous shard group s = r*8/tp .. (r+1)*8/tp-1
(unit order is shard-major, so a group is a self-contained expert of I = 256*8/tp).
  group_art(parts, man, E, ss)   pseudo-artifact of shard group ss (nq_decode can decode it; lr like nq_layer.assemble)
  kernel_expert(art, want_Q)     -> (ex, scales(lv), Q) kernel planes (+ rotated nq_decode Q2/Q4 for WDUMP checks)
  RankLayer(root, L, rank, tp)   all experts of one layer for one rank; slot_bytes() = P4 slot size (P4 + d4 + U4)
The kernel's residual code comes from proj_meta.res_rule (uniform K per projection)."""
import os,sys,json,types,torch
_R=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_R+'/threads/05-exl3-harness',_R+'/threads/12-reference-encoder'):
    if _p not in sys.path:sys.path.append(_p)
import nq_decode as D
from moe import RK_OF,RKP,rbits,proj_sizes,pack_words
PROJ=('gate','up','down')
NSH=8

def group_art(parts,man,E,ss):
    """parts[s] = torch.load(tp{s}.pt); ss contiguous shard ids."""
    art={}
    for pn in PROJ:
        m=dict(man['proj_meta'][pn]);ps=[parts[s][E][pn] for s in ss];g=len(ss)
        if pn=='down':
            m.update(k=m['k']*g//NSH,tk=m['tk']*g//NSH)
            suh2=torch.cat([p['suh2'] for p in ps]);svh2=ps[0]['svh2'];suh4=torch.cat([p['suh4'] for p in ps]);svh4=ps[0]['svh4']
        else:
            m.update(n=m['n']*g//NSH,tn=m['tn']*g//NSH)
            suh2=ps[0]['suh2'];svh2=torch.cat([p['svh2'] for p in ps]);suh4=ps[0]['suh4'];svh4=torch.cat([p['svh4'] for p in ps])
        art[pn]=dict(base=dict(shards=[p['base'] for p in ps],var=[p['var'] for p in ps],suh=suh2,svh=svh2),
                     p4=dict(shards=[p['p4'] for p in ps],word=[p['word'] for p in ps],suh=suh4,svh=svh4),meta=m)
        if 'lrU2' in ps[0]:                               # as nq_layer.assemble, restricted to the group
            if pn=='down':V,U2,U4=torch.cat([p['lrV'] for p in ps],1),ps[0]['lrU2'],ps[0]['lrU4']
            else:V=parts[ss[0]][E][ps[0].get('lrV_from',pn)]['lrV'];U2,U4=torch.cat([p['lrU2'] for p in ps],1),torch.cat([p['lrU4'] for p in ps],1)
            art[pn]['base']['lr']=dict(V=V,U2=U2);art[pn]['p4']['lr']=dict(U4=U4)
    return art

def repack(P,dev='cuda',want_Q=False):
    """(threads/13 verify_t12.repack) T12 planes -> kernel Proj fields (+ rotated nq_decode Q2/Q4 [N,K] fp16)."""
    m=P['meta'];tk,tn=m['tk'],m['tn'];U=tk*tn
    rule=m['res_rule'];assert rule['kind']=='uniform',rule
    K=float(rule['K']);rk=RK_OF[K];KA,MASK=RKP[rk];assert D.PATTERNS[K]==(KA,MASK)
    order,_=D.unit_order(tk,tn,m['shard_axis'],dev)
    Q=None
    if want_Q:
        rl=D.ring_levels(P,dev);order=rl['order']
        Q={lv:D.to_matrix(rl[f'Q{lv}'],order,tk,tn,m['k'],m['n'],dev).T.contiguous().half() for lv in (2,4)}
    rec=torch.empty(U,dtype=torch.long,device=dev);a,c=order//tn,order%tn;rec[:]=c*tk+a;inv=torch.argsort(rec)
    def lane_words(raw,bits):
        nb=8*4*bits//8;st=raw.to(dev).view(U,nb)[inv]
        b=((st.long()[...,None]>>torch.arange(8,device=dev))&1).view(U,8,4,bits).reshape(U*32,bits)
        nw=(bits+31)//32;b=torch.nn.functional.pad(b,(0,nw*32-bits)).view(U*32,nw,32)
        return (b<<torch.arange(32,device=dev)).sum(-1)
    wb=lane_words(D._cat(P['base']['shards'],'cpu'),128);wr=lane_words(D._cat(P['p4']['shards'],'cpu'),rbits(rk))
    p=types.SimpleNamespace(N=m['n'],K=m['k'],rk=rk,z=proj_sizes(m['n'],m['k'],None,rk),flags=None,fl=None)
    p.base=torch.from_numpy(wb.reshape(-1).cpu().numpy().astype('uint32').view('int32')).to(dev)
    p.p4w=wr;p.p4=pack_words(wr.cpu(),rbits(rk)).to(dev)
    bw=D._cat(P['p4']['word'],'cpu').long().to(dev)[inv];p.Mb=bw&255;p.Nn=(bw>>8)&255;p.d4=(p.Mb|(p.Nn<<8)).to(torch.int32)
    assert m['base_var']=='sign'
    v=D._cat(P['base']['var'],'cpu').long().to(dev).view(U,8)[inv];p.var=((v&1)<<torch.arange(8,device=dev)).sum(1).to(torch.uint8)
    return p,Q

def cat_proj(a,b,dev='cuda'):
    assert a.rk==b.rk,'gate and up must share the residual code (one K1 code per expert)'
    p=types.SimpleNamespace(N=a.N+b.N,K=a.K,rk=a.rk,z=proj_sizes(a.N+b.N,a.K,None,a.rk),flags=None,fl=None)
    p.base=torch.cat([a.base,b.base]);p.p4w=torch.cat([a.p4w,b.p4w]);p.p4=pack_words(p.p4w.cpu(),rbits(p.rk)).to(dev)
    p.Mb=torch.cat([a.Mb,b.Mb]);p.Nn=torch.cat([a.Nn,b.Nn]);p.d4=torch.cat([a.d4,b.d4]);p.var=torch.cat([a.var,b.var]);return p

def kernel_expert(art,dev='cuda',want_Q=False):
    g,Qg=repack(art['gate'],dev,want_Q);u,Qu=repack(art['up'],dev,want_Q);d,Qd=repack(art['down'],dev,want_Q)
    ex=types.SimpleNamespace(gu=cat_proj(g,u,dev),dn=d,H=d.N,I=d.K,lr=None,lr4=None,rg=0,rd=0)
    for p in (ex.gu,ex.dn):del p.p4w                      # only needed for cat_proj / host decode
    def lrp(pn):                                          # the encoder omits 'lr' for a projection with r = 0
        P=art[pn];n,k=P['meta']['n'],P['meta']['k'];z=lambda c:torch.zeros(0,c,dtype=torch.float16)
        if 'lr' not in P['base']:return z(k),z(n),z(n)
        return P['base']['lr']['V'].cpu(),P['base']['lr']['U2'].cpu(),P['p4']['lr']['U4'].cpu()
    (Vg,U2g,U4g),(Vu,U2u,U4u),(Vd,U2d,U4d)=(lrp(p) for p in PROJ)
    assert torch.equal(Vg,Vu),'kernel needs gate/up to share V'
    ex.rg,ex.rd=Vg.shape[0],Vd.shape[0]
    f=lambda *t:torch.cat([x.reshape(-1) for x in t]).half().contiguous().to(dev)
    if ex.rg+ex.rd:ex.lr=f(Vg,U2g,U2u,Vd,U2d);ex.lr4=f(U4g,U4u,U4d)
    def scales(lv):
        pl=D.SCALE_PLANE[lv];s=lambda n,k:art[n][pl][k].half().to(dev)
        return torch.cat([s('gate','suh'),s('gate','svh'),s('up','svh'),s('down','suh'),s('down','svh'),s('up','suh')])
    ex.sc={lv:scales(lv) for lv in (2,4)}
    return ex,scales,(Qg,Qu,Qd)

def had_width(man):
    """threads/29 in_had_down (down-projection input Hadamard width) of a layer manifest; absent = 128"""
    return int(man.get('config',{}).get('in_had_down',128))

def p4_bytes(ex):
    """bytes of the streamed level-4 part of one expert on one rank: P4 + block words (+ U4 of the lr plane)."""
    b=sum(p.p4.numel()*4+p.d4.numel()*4 for p in (ex.gu,ex.dn))
    return b+(0 if ex.lr4 is None else ex.lr4.numel()*2)

def layer_dir(root,L):
    """fit output root/L{L} or the HF layout root/layers/L{L}"""
    d=f'{root}/L{L}'
    return d if os.path.exists(f'{d}/manifest.json') else f'{root}/layers/L{L}'

def load_part(d,i):
    """tp{i}.pt, else the thread-25 safetensors container tp{i}.safetensors (same dict)"""
    if os.path.exists(f'{d}/tp{i}.pt'):return torch.load(f'{d}/tp{i}.pt',weights_only=False)
    T25=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','threads','25-campaign')
    if T25 not in sys.path:sys.path.append(T25)
    import nq25_st,struct,json,safetensors.torch as ST
    # = nq25_st.load_shard, but one sequential read: safe_open's mmap turns into small random reads (~20 MB/s on the HDD)
    b=open(f'{d}/tp{i}.safetensors','rb').read();n=struct.unpack('<Q',b[:8])[0];m=json.loads(b[8:8+n])['__metadata__']
    assert m.get('format')==nq25_st.FORMAT,m.get('format');t=ST.load(b);del b
    return nq25_st._untree(json.loads(m['tree']),t.__getitem__)

class RankLayer:
    """One MoE layer for rank `rank` of a TP=tp run, from root/L{L}/tp{s}.pt (or root/layers/L{L}/tp{s}.safetensors)."""
    def __init__(s,root,L,rank,tp=4,experts=None,dev='cuda'):
        assert NSH%tp==0 and 0<=rank<tp
        d=layer_dir(root,L);s.man=json.load(open(f'{d}/manifest.json'));assert s.man['format']=='nestquant-v1'
        g=NSH//tp;s.ss=list(range(rank*g,(rank+1)*g));s.L,s.rank,s.tp=L,rank,tp
        s.parts={i:load_part(d,i) for i in s.ss}
        s.experts=[E for E in s.man['experts'] if experts is None or E in experts];s.dev=dev
        s.ex={E:kernel_expert(group_art(s.parts,s.man,E,s.ss),dev)[0] for E in s.experts}
        s.H,s.I=s.man['proj_meta']['down']['n'],s.man['proj_meta']['down']['k']*g//NSH
        s.had_dn=had_width(s.man)                        # threads/29: down-input Hadamard width, one per layer
        assert s.I%s.had_dn==0,(L,s.I,s.had_dn)
        for x in s.ex.values():x.had_dn=s.had_dn
    def art(s,E):return group_art(s.parts,s.man,E,s.ss)
    def slot_bytes(s,align=256):
        """P4 slot size: max over this layer's experts, rounded up to `align` (the pool parameter)."""
        m=max(p4_bytes(e) for e in s.ex.values());return (m+align-1)//align*align
