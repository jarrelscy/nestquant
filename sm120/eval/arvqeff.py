"""ARVQ hybrid (GLM-5.3-Vision-NVFP4-ARVQ-hybrid) served expert weights -> dense fp32 (FP8 eval, C2).
Per MoE layer: hyb_kind[256] (2 = cold ARVQ, 0 = hot NVFP4; local index = order within its kind).
  cold: arvq-layer-LLL-{gateup,down}.safetensors  arvq_{w13,w2}_{packed,scales,codebooks,global}
        value = cold_rows(...) * global  (8+8 two-stage fp4 codebook, e4m3 scale per 16 rows x 128 cols)
  hot:  hot-layer-LLL.safetensors  nvfp4_{w13,w2}_{packed [n,k/2] U8, bscale [n,k/16] e4m3, scale2}
        value = fp4 * e4m3(bscale) * scale2 (w13: rows < n/2 use scale2[:,0] (w1 = gate), the rest scale2[:,1] (w3 = up))
cold_rows is a verbatim copy of vllm arvq_reference.cold_rows (arvq-clean-fixes12); the hot decode reads the serialized
(row-major ModelOpt) layout and is cross-checked against arvq_reference.hot_rows on the overlay's native permute in
selftest(). The served kernel also quantises activations to 4 FP4 residual planes; that is NOT modelled (weights only).
  ArvqLayer(root, L, dev).expert(e) -> (Wgate [I,H], Wup [I,H], Wdown [H,I]) fp32, kind ('hot'|'cold')"""
import os,json,torch
from safetensors import safe_open

def fp4(codes):
    lv=torch.tensor([0,0.5,1,1.5,2,3,4,6],device=codes.device,dtype=torch.float32)
    return lv[(codes&7).long()]*torch.where((codes&8)!=0,-1.,1.)
def e4m3(codes):
    codes=codes.long();assert not bool(((codes<0)|(codes>=127)).any()),'expected unsigned finite E4M3 scales'
    ex,m=codes>>3,codes&7
    return torch.where(ex==0,m.float()*(2.0**-9),(1+m.float()/8)*torch.pow(2.0,ex.float()-7))
def cold_rows(packed,codebooks,scales,n,k,expert,first=0,stop=None):
    stop=n if stop is None else stop
    assert codebooks.numel() in (384,512) and n%16==0 and k%128==0
    bits=15 if codebooks.numel()==384 else 16;device=packed.device
    row=torch.arange(first,stop,device=device)[:,None];col=torch.arange(0,k,8,device=device)[None,:]
    tile=(expert*(n//16)+row//16)*(k//64)+col//64
    position=((row%16)//8+2*((col%64)//32))*32+(row%8)*4+(col%32)//8
    bit=position*bits;word=tile*(4*bits)+bit//32;storage=packed.flatten()
    pair=(storage[word].long()&0xFFFFFFFF)>>(bit%32)
    if bits==15:pair|=(storage[word+1].long()&0xFFFFFFFF)<<(32-bit%32)
    cb=codebooks.flatten().long()&0xFFFFFFFF
    a=cb[pair&255];b=cb[256+((pair>>8)&((1<<(bits-8))-1))]
    sh=torch.arange(8,device=device)*4
    v=fp4((a[...,None]>>sh)&15)+fp4((b[...,None]>>sh)&15)
    block=scales.reshape(-1,n//16,k//128,16);sc=e4m3(block[expert,row//16,col//128,row%16])
    return (v*sc[...,None]).reshape(stop-first,k)
def hot_dense(packed,bscale):
    """serialized row-major NVFP4: packed [n,k/2] U8 (low nibble = even column), bscale [n,k/16] e4m3 -> [n,k] (no scale2)"""
    n=packed.shape[0];p=packed.long()
    v=torch.stack([fp4(p&15),fp4(p>>4)],-1).reshape(n,-1)
    return (v.view(n,-1,16)*e4m3(bscale)[...,None]).reshape(n,-1)

class ArvqLayer:
    def __init__(s,root,L,dev='cuda'):
        s.root,s.L,s.dev=root,L,dev;p=f'model.layers.{L}.mlp.experts.'
        s.fh={k:safe_open(f'{root}/{f}','pt',device='cpu') for k,f in (('gu',f'arvq-layer-{L:03d}-gateup.safetensors'),
              ('dn',f'arvq-layer-{L:03d}-down.safetensors'),('hot',f'hot-layer-{L:03d}.safetensors'))}
        s.p=p;kf=[f for f in s.fh.values() if p+'hyb_kind' in f.keys()][0]
        s.kind=kf.get_tensor(p+'hyb_kind').long();assert bool(((s.kind==0)|(s.kind==2)).all())
        s.local=torch.zeros(256,dtype=torch.long)
        for v in (0,2):m=s.kind==v;s.local[m]=torch.arange(int(m.sum()))
        s.cb={pr:s.fh[f].get_tensor(p+f'arvq_{pr}_codebooks').view(torch.int32).to(dev) for pr,f in (('w13','gu'),('w2','dn'))}
        s.alpha={pr:float(s.fh[f].get_tensor(p+f'arvq_{pr}_global')) for pr,f in (('w13','gu'),('w2','dn'))}
        s.s2={pr:s.fh['hot'].get_tensor(p+f'nvfp4_{pr}_scale2').float() for pr in ('w13','w2')}
        s.n_hot=int((s.kind==0).sum())
    @staticmethod
    def kinds(root,L):
        p=f'model.layers.{L}.mlp.experts.hyb_kind'
        for f in (f'hot-layer-{L:03d}.safetensors',f'arvq-layer-{L:03d}-gateup.safetensors'):
            F=safe_open(f'{root}/{f}','pt')
            if p in F.keys():return F.get_tensor(p).long().numpy()
    def _get(s,f,name,i):
        t=s.fh[f].get_slice(s.p+name)[i:i+1]
        return (t.view(torch.int32) if t.dtype==torch.uint32 else t).to(s.dev)
    def proj(s,e,pr):
        n,k=(4096,6144) if pr=='w13' else (6144,2048);i=int(s.local[e])
        if int(s.kind[e])==2:
            f='gu' if pr=='w13' else 'dn'
            W=cold_rows(s._get(f,f'arvq_{pr}_packed',i),s.cb[pr],s._get(f,f'arvq_{pr}_scales',i),n,k,0)*s.alpha[pr]
        else:
            W=hot_dense(s._get('hot',f'nvfp4_{pr}_packed',i)[0],s._get('hot',f'nvfp4_{pr}_bscale',i)[0])
            s2=s.s2[pr][i].to(s.dev);parts=s2.numel()
            W=(W.view(parts,n//parts,k)*s2.view(parts,1,1)).view(n,k)
        return W
    def expert(s,e):
        w13=s.proj(e,'w13');I=w13.shape[0]//2
        return w13[:I],w13[I:],s.proj(e,'w2'),('hot' if int(s.kind[e])==0 else 'cold')

def selftest(root,L=10,fp8='/data/models/zai-org/GLM-5.3'):
    import sys;sys.path.insert(0,'/home/jarrelscy/glm52/arvq-clean-fixes12/vllm/model_executor/layers/quantization')
    import importlib.util as U
    sp=U.spec_from_file_location('arvq_reference','/home/jarrelscy/glm52/arvq-clean-fixes12/vllm/model_executor/layers/quantization/arvq_reference.py')
    R=U.module_from_spec(sp);sp.loader.exec_module(R)
    A=ArvqLayer(root,L);hot=[e for e in range(256) if int(A.kind[e])==0][:2];cold=[e for e in range(256) if int(A.kind[e])==2][:2]
    for e in hot:                                   # serialized decode == overlay permute + vllm hot_rows
        for pr in ('w13','w2'):
            n,k=(4096,6144) if pr=='w13' else (6144,2048);i=int(A.local[e])
            hw=A._get('hot',f'nvfp4_{pr}_packed',i);hs=A._get('hot',f'nvfp4_{pr}_bscale',i)
            nat=hw.view(torch.int32).reshape(1,n//16,2,8,k//64,2,4).permute(0,1,4,5,2,3,6).contiguous().reshape(1,n//16,k//64,4,32)
            ref=R.hot_rows(nat,hs.view(torch.int32),n,k,0)
            got=hot_dense(hw[0],hs[0]);assert torch.equal(ref,got),('hot decode mismatch',e,pr,(ref-got).abs().max())
    print('hot serialized decode == vllm hot_rows(native permute): OK')
    sys.path.insert(0,'/data/Jarrel/nestquant/threads/18-e2e-eval');import nq_io
    F=nq_io.FP8Model(fp8)
    for e in hot+cold:
        g,u,d,kd=A.expert(e);W=F.expert(L,e,'cuda')
        r=[float((a-W[k].float()).norm()/W[k].float().norm()) for a,k in ((g,'gate_proj'),(u,'up_proj'),(d,'down_proj'))]
        sw=float((g-W['up_proj'].float()).norm()/W['up_proj'].float().norm())
        print(json.dumps(dict(L=L,e=e,kind=kd,rel_gate=round(r[0],4),rel_up=round(r[1],4),rel_down=round(r[2],4),rel_gate_vs_up=round(sw,3))))
        assert max(r)<0.5,('ARVQ vs FP8 too far (layout?)',e,r)
    print('ARVQEFF SELFTEST PASS')

if __name__=='__main__':
    import sys;torch.cuda.set_per_process_memory_fraction(float(os.environ.get('NQ_VRAM_GB','6'))/96)
    selftest(sys.argv[1] if len(sys.argv)>1 else '/data/models/jarrelscy/GLM-5.3-Vision-NVFP4-ARVQ-hybrid-pv-d4a105dc50dd',int(sys.argv[2]) if len(sys.argv)>2 else 10)
