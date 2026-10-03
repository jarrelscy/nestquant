"""Merge a TP4 repack (repack.py layout: rank{r}.json/.bin + res/rank{r}/L{L}.pt, the HF release repo root) into TP2
rank r from TP4 ranks 2r, 2r+1, byte-identical to `repack.py ROOT OUT 2` from the encode shards (2x DGX Spark: each node
downloads only its two TP4 rank files).
  tp4to2.py IN OUT r [layers=3-77] [rec_bytes]
Plane relations (sm120/nqload.py): kernel unit order is strip-major, chunk-minor (16-row strip s, 128-col chunk c,
record = (s*C + c)*32 + lane); gate|up strips are N-sharded (TP2 gu = gate_2r | gate_2r+1 | up_2r | up_2r+1), down chunks
K-sharded (TP2 chunk = half*C4 + c). The packed sub-arrays (uint4 x n4 | uint2 | uint | ushort | bit-packed tail,
moe.pack_words) are permuted per record without decoding. Scale vectors [H su_g | I sv_g | I sv_u | I su_d | H sv_o |
H su_u]: the I parts concatenate, the H parts are equal in both halves. Low-rank: V_g, U_d (U2, U4) are equal in both
halves; U_g, U_u (U2, U4) and V_d concatenate. Every equality is asserted."""
import os,sys,json,types,fcntl,time,torch
torch.set_num_threads(int(os.environ.get('NT','8')))
HERE=os.path.dirname(os.path.abspath(__file__));sys.path[:0]=[HERE,HERE+'/../sm120']
if 'build' not in sys.modules:                           # moe imports the kernel at import time; the merge is host-only
    sys.modules['build']=types.SimpleNamespace(get=lambda *a,**k:None,get_sal=lambda *a,**k:None)
from moe import rbits,split_bits
import p4rec as PR,resident as RS
from repack import parse_layers,NE,L0

def split(t,nrec,bits):
    """packed sub-array int32 tensor -> list of per-record uint8 parts ([nrec, nb] bytes; tail [nrec, T] bits)"""
    n4,n2,n1,nh,T=split_bits(bits);b=t.contiguous().view(torch.uint8);o=0;out=[]
    for nb in (16*n4,8*n2,4*n1,2*nh):
        if nb:out.append(b[o:o+nrec*nb].view(nrec,nb));o+=nrec*nb
    if T:
        bb=b[o:o+(nrec*T+7)//8].long()
        out.append(((bb[:,None]>>torch.arange(8))&1).reshape(-1)[:nrec*T].view(nrec,T).to(torch.uint8))
    return out

def join(parts,bits):
    """inverse of split, = moe.pack_words (tail bit-packed over records + 32 zero bits, cut to whole bytes)"""
    T=split_bits(bits)[4];out=[p.reshape(-1) for p in (parts[:-1] if T else parts)]
    if T:
        b=torch.cat([parts[-1].reshape(-1),torch.zeros(32,dtype=torch.uint8)]);b=b[:b.numel()//8*8].view(-1,8).long()
        out.append((b<<torch.arange(8)).sum(1).to(torch.uint8))
    return torch.cat(out).view(torch.int32)

def perm_units(S,C,half_gu):
    """TP2 unit -> unit index in cat(rank 2r units, rank 2r+1 units). gu: S = TP4 strips (gate S/2 | up S/2); dn: C = TP4 chunks"""
    if half_gu:
        n=S*C;g=S//2*C;a=torch.arange(g)
        return torch.cat([a,n+a,g+a,n+g+a])
    s=torch.arange(S)[:,None,None];h=torch.arange(2)[None,:,None];c=torch.arange(C)[None,None,:]
    return (h*S*C+s*C+c).reshape(-1)

def merge_proj(pa,pb,u,nrec_unit=32):
    """pa, pb: TP4 projections (base, var, p4, d4 int32 tensors, rk, bk) -> TP2 projection; u = perm_units"""
    assert pa.rk==pb.rk and pa.bk==pb.bk,('rk/bk differ between the TP4 halves',pa.rk,pb.rk,pa.bk,pb.bk)
    U=pa.d4.numel();rec=(u[:,None]*nrec_unit+torch.arange(nrec_unit)).reshape(-1)
    def sub(ta,tb,bits):
        A,B=split(ta,U*nrec_unit,bits),split(tb,U*nrec_unit,bits)
        assert torch.equal(join(A,bits),ta) and torch.equal(join(B,bits),tb),'sub-array round trip'
        return join([torch.cat([x,y])[rec] for x,y in zip(A,B)],bits)
    p=types.SimpleNamespace(rk=pa.rk,bk=pa.bk,p4=None,d4=None,flags=None,fl=None)
    p.base=sub(pa.base,pb.base,rbits(pa.bk));p.p4=sub(pa.p4,pb.p4,rbits(pa.rk))
    p.d4=torch.cat([pa.d4,pb.d4])[u].contiguous();p.var=torch.cat([pa.var,pb.var])[u].contiguous()
    return p

def eq(a,b,what):
    assert torch.equal(a,b),f'{what} differs between the TP4 halves'
    return a

def merge_vec(a,b,H,I):
    """[H su_g | I sv_g | I sv_u | I su_d | H sv_o | H su_u]"""
    sa,sb=a.split([H,I,I,I,H,H]),b.split([H,I,I,I,H,H])
    return torch.cat([eq(sa[0],sb[0],'su_g'),sa[1],sb[1],sa[2],sb[2],sa[3],sb[3],eq(sa[4],sb[4],'sv_o'),eq(sa[5],sb[5],'su_u')])

def merge_lr(a,b,H,I,rg,rd):
    """lr = Vg [rg,H] | U2g [rg,I] | U2u [rg,I] | Vd [rd,I] | U2d [rd,H]"""
    sz=[rg*H,rg*I,rg*I,rd*I,rd*H];A,B=a.split(sz),b.split(sz);c1=lambda x,y,r:torch.cat([x.view(r,I),y.view(r,I)],1).reshape(-1)
    return torch.cat([eq(A[0],B[0],'lr V_g'),c1(A[1],B[1],rg),c1(A[2],B[2],rg),c1(A[3],B[3],rd),eq(A[4],B[4],'lr U2_d')])

def merge_lr4(a,b,H,I,rg,rd):
    """lr4 = U4g [rg,I] | U4u [rg,I] | U4d [rd,H]"""
    sz=[rg*I,rg*I,rd*H];A,B=a.split(sz),b.split(sz);c1=lambda x,y,r:torch.cat([x.view(r,I),y.view(r,I)],1).reshape(-1)
    return torch.cat([c1(A[0],B[0],rg),c1(A[1],B[1],rg),eq(A[2],B[2],'lr U4_d')])

class Src:
    """one TP4 rank: index, record file, one layer's res"""
    def __init__(s,root,r):
        s.r=r;s.idx=json.load(open(f'{root}/rank{r}.json'));s.root=root;s.rb=s.idx['rec_bytes']
        assert s.idx['tp']==4 and s.idx['rank']==r and s.idx['L0']==L0 and s.idx['NE']==NE,(r,{k:v for k,v in s.idx.items() if k!='layers'})
        s.fd=os.open(f'{root}/rank{r}.bin',os.O_RDONLY)
    def layer(s,L):
        d=torch.load(f'{s.root}/res/rank{s.r}/L{L}.pt',map_location='cpu',weights_only=False)
        assert d['format'] in ('nq-res-v1','nq-res-v2') and d['tp']==4 and d['rank']==s.r and d['L']==L,(d['format'],d['tp'],d['rank'],d['L'])
        return d
    def record(s,L,E):
        b=os.pread(s.fd,s.rb,((L-L0)*NE+E)*s.rb);assert len(b)==s.rb,('short read',s.r,L,E)
        rb=torch.frombuffer(bytearray(b),dtype=torch.uint8)
        return {k:rb[o:o+n].view(torch.int32) if k!='lr4' else rb[o:o+n].view(torch.float16) for k,(o,n) in s.idx['seg'].items()}

def merge_layer(A,B,L,r):
    da,db=A.layer(L),B.layer(L);H,I4=da['H'],da['I'];I=2*I4;n=len(da['experts'])
    assert db['H']==H and db['I']==I4 and da['experts']==db['experts']
    for k in ('rk_gu','rk_dn','rg','rd','bk_gu','bk_dn'):assert da.get(k)==db.get(k),(L,k)
    w=da.get('in_had_down',128);assert db.get('in_had_down',128)==w
    bkg,bkd=da.get('bk_gu',[0]*n),da.get('bk_dn',[0]*n)
    ugu,udn=perm_units(2*I4//16,H//128,True),perm_units(H//16,I4//128,False)
    ex={}
    for i,E in enumerate(da['experts']):
        ra,rb=A.record(L,E),B.record(L,E);rg,rd=da['rg'][i],da['rd'][i];has=rg+rd>0
        P=lambda d,rec,k,rk,bk:types.SimpleNamespace(base=d[k+'_base'][i],var=d[k+'_var'][i],p4=rec[k+'.p4'],d4=rec[k+'.d4'],rk=rk,bk=bk)
        gu=merge_proj(P(da,ra,'gu',da['rk_gu'][i],bkg[i]),P(db,rb,'gu',da['rk_gu'][i],bkg[i]),ugu)
        dn=merge_proj(P(da,ra,'dn',da['rk_dn'][i],bkd[i]),P(db,rb,'dn',da['rk_dn'][i],bkd[i]),udn)
        x=types.SimpleNamespace(gu=gu,dn=dn,H=H,I=I,rg=rg,rd=rd,lr=None,lr4=None,had_dn=w,
                                sc={lv:merge_vec(da[f'sc{lv}'][i],db[f'sc{lv}'][i],H,I4) for lv in (2,4)})
        if has:
            n2,n4=RS.lr_len(H,I4,rg,rd),(2*I4*rg+H*rd)
            for d in (da,db):assert not d['lr'][i,n2:].any(),('lr padding not zero',L,E)
            for rec in (ra,rb):assert not rec['lr4'][n4:].any(),('lr4 padding not zero',L,E)
            x.lr=merge_lr(da['lr'][i,:n2],db['lr'][i,:n2],H,I4,rg,rd);x.lr4=merge_lr4(ra['lr4'][:n4],rb['lr4'][:n4],H,I4,rg,rd)
        else:
            for rec in (ra,rb):assert not rec['lr4'].any()
        ex[E]=x
    return types.SimpleNamespace(L=L,rank=r,tp=2,H=H,I=I,experts=list(da['experts']),ex=ex,had_dn=w)

def main():
    src,out,r=sys.argv[1],sys.argv[2],int(sys.argv[3])
    layers=parse_layers(sys.argv[4]) if len(sys.argv)>4 else list(range(3,78))
    A,B=Src(src,2*r),Src(src,2*r+1);assert A.idx['seg']==B.idx['seg'] and A.rb==B.rb
    os.makedirs(out,exist_ok=True)
    for L in layers:
        if str(L) not in A.idx['layers'] or str(L) not in B.idx['layers']:print(f'L{L}: not in the TP4 index');continue
        assert A.idx['layers'][str(L)]==B.idx['layers'][str(L)],('index entries differ',L)
        ip=f'{out}/rank{r}.json';idx=json.load(open(ip)) if os.path.exists(ip) else dict(format='nq-p4rec-v1',tp=2,rank=r,L0=L0,NE=NE,layers={})
        rp=f'{out}/res/rank{r}/L{L}.pt';os.makedirs(os.path.dirname(rp),exist_ok=True)
        if str(L) in idx['layers'] and os.path.exists(rp):continue
        t=time.time();RL=merge_layer(A,B,L,r)
        if not os.path.exists(rp):RS.save(RL,rp);print(f'L{L} rank{r}: resident planes {os.path.getsize(rp)/2**20:.0f} MiB',flush=True)
        if str(L) in idx['layers']:continue
        lay=PR.layout(next(iter(RL.ex.values())),RL.H,RL.I)
        if 'rec_bytes' not in idx:idx['rec_bytes']=int(sys.argv[5]) if len(sys.argv)>5 else lay['rec_bytes'];idx['seg']=json.loads(json.dumps(lay['seg']))
        assert json.loads(json.dumps(lay['seg']))==idx['seg'] and lay['rec_bytes']<=idx['rec_bytes'],('layout changed',L,lay,idx['seg'])
        lay['rec_bytes']=idx['rec_bytes'];rb=lay['rec_bytes']
        fd=os.open(f'{out}/rank{r}.bin',os.O_WRONLY|os.O_CREAT,0o644)
        try:
            for E,x in RL.ex.items():os.pwrite(fd,PR.pack(x,lay),((L-L0)*NE+E)*rb)
            os.fsync(fd)
        finally:os.close(fd)
        ent=dict(experts=RL.experts,rg={E:RL.ex[E].rg for E in RL.experts},rd={E:RL.ex[E].rd for E in RL.experts})
        with open(f'{out}/rank{r}.lock','w') as lk:
            fcntl.flock(lk,fcntl.LOCK_EX)
            if os.path.exists(ip):cur=json.load(open(ip));cur.setdefault('rec_bytes',idx['rec_bytes']);cur.setdefault('seg',idx['seg']);idx=cur
            idx['layers'][str(L)]=ent
            json.dump(idx,open(f'{ip}.{os.getpid()}.tmp','w'));os.replace(f'{ip}.{os.getpid()}.tmp',ip)
        print(f'L{L} rank{r}: {len(RL.experts)} records x {rb} B in {time.time()-t:.1f}s',flush=True)

if __name__=='__main__':main()
