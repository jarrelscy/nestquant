import os,sys,json,torch,torch.nn.functional as F
os.environ.setdefault('OMP_NUM_THREADS','8');torch.set_num_threads(8)
sys.path.insert(0,'/home/coder/git/orbit-duet')
OD='/home/coder/git/orbit-duet'
SRC='/tmp/orbit-duet-glm53-fp8'
RUN=f'{OD}/runs/glm53_pilot_matched_l16'
CAP=f'{OD}/runs/glm53_matched_context_pilot_v1_capture/layer_16.pt'
SCR='/tmp/nestquant/08-ood-robustness'
def init():
    torch.cuda.set_per_process_memory_fraction(12/80);torch.backends.cuda.matmul.allow_tf32=False
def native(E):
    p=f'{SCR}/w_e{E}.pt'
    if os.path.exists(p):return [w.cuda() for w in torch.load(p)]
    from orbit_duet.source import weights
    from orbit_duet.statistics import tensor_hash
    w=weights(SRC,16,E);meta=json.load(open(f'{RUN}/statistics/l16_e{E}.json'))
    assert [tensor_hash(v) for v in w]==meta['teacher_tensor_sha256'],'teacher hash'
    torch.save([v.cpu() for v in w],p);return w
def teacher(x,w):
    g,u,d=w
    return F.linear(F.silu(F.linear(x.bfloat16(),g.bfloat16()))*F.linear(x.bfloat16(),u.bfloat16()),d.bfloat16()).float()
def hidden(x,w):
    g,u,d=w
    return (F.silu(F.linear(x.bfloat16(),g.bfloat16()))*F.linear(x.bfloat16(),u.bfloat16())).float()
def sample(E):
    t=torch.load(f'{RUN}/statistics/l16_e{E}_training_sample.pt',weights_only=True,mmap=True)
    return t['x'],t['p']
def stats(E):
    return torch.load(f'{RUN}/statistics/l16_e{E}.pt',weights_only=True,mmap=True)
@torch.no_grad()
def grams(x,wts,w,bs=2048):
    """sum wts^2 x x^T and h h^T (matching stats recipe: value = x*p)"""
    G=torch.zeros(6144,6144,device='cuda',dtype=torch.float64);D=torch.zeros(2048,2048,device='cuda',dtype=torch.float64)
    for i in range(0,len(x),bs):
        xx=x[i:i+bs].cuda();pp=wts[i:i+bs].cuda().float()[:,None]
        h=hidden(xx,w)
        a=(xx.float()*pp);G.addmm_(a.T.double(),a.double()) if False else G.add_((a.T@a).double())
        b=h*pp;D.add_((b.T@b).double())
    return G.float(),D.float()
@torch.no_grad()
def rel_err(x,w_nat,w_q,bs=1024,weights=None):
    num=den=0.
    for i in range(0,len(x),bs):
        xx=x[i:i+bs].cuda();t=teacher(xx,w_nat).double();q=teacher(xx,w_q).double()
        e=(q-t).square().sum(-1);en=t.square().sum(-1)
        if weights is not None:ww=weights[i:i+bs].cuda().double()**2;e=e*ww;en=en*ww
        num+=float(e.sum());den+=float(en.sum())
    return (num/den)**.5*100
def capture():
    c=torch.load(CAP,weights_only=True,mmap=True)
    groups={}
    for di,d in enumerate(c['domains']):groups.setdefault(d,[]).append(di)
    rows={}
    for d,docs in groups.items():rows[d]=torch.isin(c['document_ids'],torch.tensor(docs)).nonzero().flatten()
    rows['ID']=torch.cat([v for k,v in rows.items() if k.startswith('control:')]).sort().values
    rows['OOD']=torch.cat([v for k,v in rows.items() if k.startswith('ood:')]).sort().values
    return c,rows
def exl3_fit(w,HG,HD,bits,count,sigma=.03,seed=91426):
    from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3,get_temp_buffers
    out=[]
    for i,weight in enumerate(w):
        sg=sigma[int(i==2)] if isinstance(sigma,(tuple,list)) else sigma
        qa=dict(K=bits,devices=['cuda:0'],seed=seed,sigma_reg=sg,apply_out_scales=None,mul1=True)
        hd=dict(H=(HD if i==2 else HG).clone().cuda(),count=count,finalized=False,device=torch.device('cuda:0'))
        with torch.no_grad():wq,proxy,val=quantize_exl3(weight.T.contiguous(),hd,qa,False,verbose=False)
        out.append(wq.T.contiguous().float());get_temp_buffers.cache_clear();del hd,val;torch.cuda.empty_cache()
    return out
def exl3_artifact(E,bits):
    from orbit_duet.exl3_adapter import EXL3Expert
    return EXL3Expert(f'{RUN}/exl3_e{E}/expert_{bits}.bin').decoded_weights()
def nvfp4(E,w):
    from orbit_duet.modelopt_nvfp4 import load_artifact
    s=json.load(open(f'{RUN}/statistics/l16_e{E}.json'))
    payload=torch.load(f'{RUN}/nvfp4_e{E}/weights.pt',weights_only=True,mmap=True)
    ident=payload['metadata']['training_statistics']
    return load_artifact(f'{RUN}/nvfp4_e{E}/weights.pt',w,16,E,ident)[0]
