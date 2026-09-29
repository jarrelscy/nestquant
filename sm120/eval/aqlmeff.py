"""AQLM hybrid (GLM-5.3-Vision-NVFP4-AQLM-hybrid-1m, quant_method nvfp4_aqlm_hybrid) served expert weights -> dense
fp32 (FP8 eval, C2). Weights only, like arvqeff.py.
Per MoE layer (all tensors in the sharded model-*.safetensors, found through model.safetensors.index.json):
  hyb_kind [256] int8: 0 = hot NVFP4, 1 = base AQLM (w13 1x16, w2 2x16), 2 = cold AQLM (w13 1x16, w2 1x16).
  Experts are packed per group in ascending global id: hot -> nvfp4_*[i], w13 (base + cold) -> w13_*[i],
  w2 base -> w2m_*[i], w2 cold -> w2c_*[i] (i = rank of the expert within its group). This checkpoint has n_base = 0.
  AQLM: codes int16 [n, books, rows, cols/8] (read as uint16), codebooks fp16 [books, 65536, 8], scales fp16 [n, rows]
        value = fp16( fp16(sum over books of codebook[b][code]) * scale[row] )
  hot:  nvfp4_{w13,w2}_packed [n, rows, cols/2] U8 (low nibble = even column), _bscale [n, rows, cols/16] e4m3,
        _scale2 f32 (w13 [n,2] = gate, up; w2 [n,1]); value = fp16( fp4 * fp32(e4m3(bscale) * scale2[part]) )
Both formulas are the vLLM prefill dequant kernels verbatim (csrc/quantization/aqlm_moe/aqlm_moe_v2.cu:
CodeKx16DequantMoE, NvFp4DequantMoE; the serving tree arvq-clean-fixes12). The decode gemv kernels accumulate the same
fp16 weights against fp16 activations; activation rounding is NOT modelled (weights only). selftest() checks the cold
path bitwise against nvfp4_aqlm_hybrid._dequant_reference (loaded from the serving tree's source), the hot path against
the ARVQ checkpoint's hot experts (same donor NVFP4 bytes) and every expert for layout sanity against FP8.
  AqlmLayer(root, L, dev).expert(e) -> (Wgate [I,H], Wup [I,H], Wdown [H,I]) fp32, kind ('hot'|'base'|'cold')
  AqlmLayer.kinds(root, L) -> hyb_kind numpy (0 hot)"""
import os,json,torch
import numpy as np
from safetensors import safe_open

AQLM='/data/huggingface/hub/models--jarrelscy--GLM-5.3-Vision-NVFP4-AQLM-hybrid-1m/snapshots/2b883d28bb9dd13a9511e2bd45a8ad1cbacbad74'
VLLM_TREE='/home/jarrelscy/glm52/arvq-clean-fixes12'
FP4=torch.tensor([0,0.5,1,1.5,2,3,4,6,-0.,-0.5,-1,-1.5,-2,-3,-4,-6],dtype=torch.float32)
_IX={}

def _index(root):
    if root not in _IX:_IX[root]=json.load(open(f'{root}/model.safetensors.index.json'))['weight_map']
    return _IX[root]
def e4m3(u8):return u8.contiguous().view(torch.float8_e4m3fn).float()
def aqlm_rows(codes,codebooks,scales):
    """codes [books, rows, cols/8] int16, codebooks [books, 65536, 8] fp16, scales [rows] fp16 -> [rows, cols] fp16"""
    idx=codes.view(torch.uint16).long() if codes.dtype==torch.int16 else codes.long()
    w=codebooks[0][idx[0]]
    for b in range(1,idx.shape[0]):w=w+codebooks[b][idx[b]]        # fp16 adds (__hadd2)
    return (w.reshape(idx.shape[1],-1)*scales[:,None])               # fp16 multiply (__hmul2), round to nearest
def nvfp4_rows(packed,bscale,s2):
    """packed [rows, cols/2] u8, bscale [rows, cols/16] u8 (e4m3), s2 [parts] f32 (rows split evenly) -> [rows, cols] fp16"""
    n=packed.shape[0];p=packed.long();lut=FP4.to(packed.device)
    v=torch.stack([lut[p&15],lut[p>>4]],-1).reshape(n,-1,16)         # fp32
    part=(torch.arange(n,device=packed.device)*s2.numel())//n
    sc=e4m3(bscale)*s2.to(packed.device)[part][:,None]                  # fp32 e4m3 * gscale
    return (v*sc[...,None]).reshape(n,-1).half()

class AqlmLayer:
    def __init__(s,root,L,dev='cuda'):
        s.root,s.L,s.dev=root,L,dev;s.p=f'model.layers.{L}.mlp.experts.';ix=_index(root);s.fh={}
        s.kind=s._t('hyb_kind').long().cpu();assert bool(((s.kind>=0)&(s.kind<=2)).all())
        def lk(m):return torch.where(m,torch.cumsum(m.long(),0)-1,torch.full_like(s.kind,-1))
        s.i_nv,s.i_13,s.i_2m,s.i_2c=lk(s.kind==0),lk(s.kind!=0),lk(s.kind==1),lk(s.kind==2)
        s.cb={n:s._t(n+'_codebooks').to(dev) for n in ('w13','w2m','w2c') if s.p+n+'_codebooks' in ix}
        s.s2={pr:s._t(f'nvfp4_{pr}_scale2').float() for pr in ('w13','w2')} if s.p+'nvfp4_w13_scale2' in ix else {}
        s.n_hot=int((s.kind==0).sum())
    def _f(s,name):
        f=_index(s.root)[s.p+name]
        if f not in s.fh:s.fh[f]=safe_open(f'{s.root}/{f}','pt',device='cpu')
        return s.fh[f]
    def _t(s,name):return s._f(name).get_tensor(s.p+name)
    def _row(s,name,i):return s._f(name).get_slice(s.p+name)[i:i+1][0].to(s.dev)
    @staticmethod
    def kinds(root,L):
        p=f'model.layers.{L}.mlp.experts.hyb_kind';return safe_open(f'{root}/{_index(root)[p]}','pt').get_tensor(p).long().numpy()
    def proj16(s,e,pr):
        """fp16 dequant of one projection of global expert e (w13 [2I,H] gate then up, w2 [H,I])"""
        k=int(s.kind[e])
        if k==0:
            i=int(s.i_nv[e]);return nvfp4_rows(s._row(f'nvfp4_{pr}_packed',i),s._row(f'nvfp4_{pr}_bscale',i),s.s2[pr][i])
        if pr=='w13':n,i='w13',int(s.i_13[e])
        else:n,i=('w2m',int(s.i_2m[e])) if k==1 else ('w2c',int(s.i_2c[e]))
        return aqlm_rows(s._row(n+'_codes',i),s.cb[n],s._row(n+'_scales',i))
    def expert(s,e):
        w13=s.proj16(e,'w13').float();I=w13.shape[0]//2
        return w13[:I],w13[I:],s.proj16(e,'w2').float(),{0:'hot',1:'base',2:'cold'}[int(s.kind[e])]
    def bits(s):
        """bits per weight of one expert: cold = codes + scales (+ its share of the layer's codebooks), hot = packed +
        bscale + scale2; from the checkpoint tensor shapes."""
        P=3*2048*6144;sh=lambda n:int(np.prod(s._f(n).get_slice(s.p+n).get_shape()))
        nc=int((s.kind==2).sum());nh=s.n_hot;out={}
        if nc:
            cold=(sh('w13_codes')*2+sh('w13_scales')*2)/int((s.kind!=0).sum())+(sh('w2c_codes')*2+sh('w2c_scales')*2)/nc
            out.update(cold=8*cold/P,cold_books=8*(cold+(sh('w13_codebooks')+sh('w2c_codebooks'))*2/nc)/P)
        if nh:out['hot']=8*((sh('nvfp4_w13_packed')+sh('nvfp4_w13_bscale')+sh('nvfp4_w2_packed')+sh('nvfp4_w2_bscale'))/nh+12)/P
        return out

def _vllm_dequant_reference():
    """nvfp4_aqlm_hybrid._dequant_reference, compiled from the serving tree's source (the module itself needs vllm)."""
    import ast
    src=open(f'{VLLM_TREE}/vllm/model_executor/layers/quantization/nvfp4_aqlm_hybrid.py').read()
    fn=[n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name=='_dequant_reference'][0]
    g={'torch':torch};exec(compile(ast.Module([fn],[]),'nvfp4_aqlm_hybrid.py','exec'),g);return g['_dequant_reference']

def selftest(root=AQLM,L=10,fp8='/data/models/zai-org/GLM-5.3',arvq='/data/models/jarrelscy/GLM-5.3-Vision-NVFP4-ARVQ-hybrid-pv-d4a105dc50dd',dev='cpu'):
    import sys
    ref=_vllm_dequant_reference();A=AqlmLayer(root,L,dev)
    hot=[e for e in range(256) if int(A.kind[e])==0];cold=[e for e in range(256) if int(A.kind[e])==2]
    print(json.dumps(dict(L=L,n_hot=len(hot),n_cold=len(cold),bits=A.bits())))
    for e in cold[:2]+cold[-1:]:                   # cold == vLLM _dequant_reference, bitwise
        for n,i in (('w13',int(A.i_13[e])),('w2c',int(A.i_2c[e]))):
            c=A._f(n+'_codes').get_slice(A.p+n+'_codes')[i:i+1];sc=A._f(n+'_scales').get_slice(A.p+n+'_scales')[i:i+1]
            r=ref(c,A.cb[n].cpu(),sc)[0];got=A.proj16(e,'w13' if n=='w13' else 'w2').cpu()
            assert r.dtype==torch.float16 and torch.equal(r.view(torch.int16),got.view(torch.int16)),('cold mismatch',e,n)
    print('cold AQLM == vllm _dequant_reference (bitwise): OK')
    import arvqeff                                 # hot: same donor NVFP4 bytes as the ARVQ checkpoint's hot experts
    R=arvqeff.ArvqLayer(arvq,L,dev);both=[e for e in hot if int(R.kind[e])==0][:3];assert both,'no expert hot in both'
    for e in both:
        for pr in ('w13','w2'):
            i,j=int(A.i_nv[e]),int(R.local[e])
            for t in ('packed','bscale'):
                a_=A._row(f'nvfp4_{pr}_{t}',i);b_=R._get('hot',f'nvfp4_{pr}_{t}',j)[0]
                assert torch.equal(a_.view(torch.uint8),b_.view(torch.uint8)),('hot bytes differ from ARVQ hot',e,pr,t)
            assert torch.equal(A.s2[pr][i],R.s2[pr][j]),('scale2 differ',e,pr)
            W=R.proj(e,pr).cpu();got=A.proj16(e,pr).float().cpu()
            rel=float((W-got).norm()/W.norm());assert rel<1e-3,('hot decode differs from arvqeff beyond fp16 rounding',e,pr,rel)
    print(f'hot NVFP4 bytes == ARVQ checkpoint hot experts {both}, dequant == arvqeff.hot within fp16 rounding: OK')
    sys.path.insert(0,'/data/Jarrel/nestquant/threads/18-e2e-eval');import nq_io
    F=nq_io.FP8Model(fp8)
    for e in hot[:2]+cold[:3]:
        g,u,d,kd=A.expert(e);W=F.expert(L,e,dev)
        r=[float((a-W[k].float()).norm()/W[k].float().norm()) for a,k in ((g,'gate_proj'),(u,'up_proj'),(d,'down_proj'))]
        sw=float((g-W['up_proj'].float()).norm()/W['up_proj'].float().norm())
        print(json.dumps(dict(L=L,e=e,kind=kd,rel_gate=round(r[0],4),rel_up=round(r[1],4),rel_down=round(r[2],4),rel_gate_vs_up=round(sw,3))))
        assert max(r)<0.8 and sw>1.0,('AQLM vs FP8 too far (layout?)',e,r,sw)
    print('AQLMEFF SELFTEST PASS')

if __name__=='__main__':
    import sys
    selftest(L=int(sys.argv[1]) if len(sys.argv)>1 else 10,dev=sys.argv[2] if len(sys.argv)>2 else 'cpu')
