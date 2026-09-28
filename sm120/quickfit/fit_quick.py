"""Quick weights-only NestQuant fit of GLM-5.3 routed experts (shape/speed testing, not accuracy).
H = identity (no calibration statistics, so LDLQ feedback is a no-op), G = None, inner = 0, single pass,
residual K from NQ_RES_K (default '2,2,2' = uniform K2 via exllamav3's CUDA Viterbi, 4.02 bpw; the production pattern
'1.9375,1.9375,2.3125' = 4.0846 bpw goes through the torch pattern Viterbi, ~80 s/expert)."""
import os,sys,json,time,torch
R='/data/Jarrel/nestquant'
os.environ.setdefault('CUDA_HOME','/home/jarrelscy/cuda128');os.environ['TORCH_CUDA_ARCH_LIST']='12.0a'
os.environ['PATH']=os.environ['CUDA_HOME']+'/bin:'+os.path.dirname(sys.executable)+':'+os.environ['PATH']
sys.path[:0]=[R+'/threads/12-reference-encoder',R+'/threads/05-exl3-harness']
from safetensors import safe_open
import nq_encode as NE, nq_patvit as PV, functools
PCHUNK=int(os.environ.get('NQ_PAT_CHUNK','1024'))
_patq=PV.patq
PV.patq=lambda tiles,K,chunk=None:_patq(tiles,K,chunk=PCHUNK)   # same Viterbi, bigger batches (bit-identical states)
RES_K=dict(zip(NE.PROJ,map(float,os.environ.get('NQ_RES_K','2,2,2').split(','))))
NE.RES_KS=()                                                    # g-scale search only for the residual K in use
SRC='/data/models/zai-org/GLM-5.3'
WM=json.load(open(SRC+'/model.safetensors.index.json'))['weight_map']
_fh={}
def T(name):
    f=WM[name]
    if f not in _fh:_fh[f]=safe_open(SRC+'/'+f,'pt',device='cpu')
    return _fh[f].get_tensor(name)
def fp8_dequant(name):
    w=T(name).to('cuda',torch.float32);s=T(name+'_scale_inv').to('cuda',torch.float32)
    o,i=w.shape;return (w.view(o//128,128,i//128,128)*s[:,None,:,None]).view(o,i)
def teacher(L,E):
    p=f'model.layers.{L}.mlp.experts.{E}.';return [fp8_dequant(p+k+'_proj.weight') for k in ('gate','up','down')]
def fit(L,E,check=False):
    Ws=teacher(L,E)
    HG={'H':[torch.eye(Ws[0].shape[1],device='cuda'),torch.eye(Ws[0].shape[1],device='cuda'),torch.eye(Ws[2].shape[1],device='cuda')],'G':[None,None,None]}
    return NE.encode_expert(Ws,HG,count=1,base_var=None,inner=0,check=check,canonical_base=False,res_K=RES_K)
if __name__=='__main__':
    L,E=int(sys.argv[1]),int(sys.argv[2])
    torch.backends.cuda.matmul.allow_tf32=False
    t=time.time();art,dense=fit(L,E,check=True);dt=time.time()-t
    Ws=teacher(L,E)
    for i,pn in enumerate(NE.PROJ):
        for lv in (2,4):
            W=Ws[i].cpu();Q=dense[lv][i];print(pn,lv,'rel err',round(float((Q-W).norm()/W.norm()),4))
    print('meta',{p:art['meta']['info'][p].get('bits') for p in NE.PROJ},'rate',art['meta']['rate'],'sec',round(dt,1))
