"""Create a separate Flash backbone overlay with block-FP8 attention and head.
The published source checkpoint is never edited. Routed NQ planes stay bit-exact.
"""
import argparse,json,os,re
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file


def target(name,config):
    if name=='lm_head.weight':return True
    m=re.search(r'layers\.(\d+)\.self_attn\.(\w+)\.weight$',name)
    if not m:return False
    L,proj=int(m[1]),m[2]
    kinds=config['text_config']['layer_types']
    kda=L<len(kinds) and kinds[L]=='linear_attention'
    return proj in ('q_proj','k_proj','v_proj','o_proj') if kda else proj=='kv_b_proj'


def quantize(w):
    n,k=w.shape
    if k%128:raise ValueError('FP8 input dimension must be divisible by 128')
    # Scales use max-abs/448, e4m3 finite range; tail rows are zero padded.
    q=torch.empty((n,k),dtype=torch.float8_e4m3fn)
    scales=torch.empty(((n+127)//128,k//128),dtype=torch.float32)
    for row in range(0,n,128):
        block=w[row:row+128].float()
        if block.shape[0]<128:block=torch.nn.functional.pad(block,(0,0,0,128-block.shape[0]))
        tiles=block.reshape(128,k//128,128)
        scale=tiles.abs().amax((0,2)).clamp_min(1e-12)/448
        q[row:row+128]=(tiles/scale[None,:,None]).clamp(-448,448).reshape(128,k)[:min(128,n-row)].to(q.dtype)
        scales[row//128]=scale
    return q,scales


def convert(src,dst):
    src=Path(src).resolve();dst=Path(dst).resolve()
    if src==dst:raise ValueError('Output must differ from checkpoint')
    dst.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((src/'config.json').read_text())
    index=json.loads((src/'model.safetensors.index.json').read_text())
    files=sorted({v for k,v in index['weight_map'].items() if target(k,cfg)})
    changed=[]
    for file in files:
        marker=dst/(file+'.complete.json')
        if marker.exists():
            names=json.loads(marker.read_text());changed+=names
            for name in names:index['weight_map'][name.removesuffix('.weight')+'.weight_scale_inv']=file
            continue
        tensors={};names=[]
        with safe_open(src/file,framework='pt',device='cpu') as f:
            for name in f.keys():
                w=f.get_tensor(name)
                if target(name,cfg) and w.dtype in (torch.bfloat16,torch.float16,torch.float32):
                    q,scale=quantize(w);tensors[name]=q
                    sn=name.removesuffix('.weight')+'.weight_scale_inv'
                    tensors[sn]=scale;index['weight_map'][sn]=file;names.append(name)
                else:tensors[name]=w
            save_file(tensors,str(dst/(file+'.tmp')))
        os.replace(dst/(file+'.tmp'),dst/file)
        marker.write_text(json.dumps(names,indent=2)+'\n');changed+=names
        del tensors
        print(f'{file}: converted {len(names)} tensors',flush=True)
    for file in src.iterdir():
        if file.name in ('config.json','model.safetensors.index.json') or file.name in files or file.name.startswith('.'):continue
        dest=dst/file.name
        if not dest.exists():dest.symlink_to(file,target_is_directory=file.is_dir())
    cfg['nestquant_flash']={'native_fp8_attention':True,'conversion':'e4m3fn maxabs/448, block128, zero-pad tail rows','quality':'published research reference; not remeasured here'}
    (dst/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    index.pop('metadata',None)
    (dst/'model.safetensors.index.json').write_text(json.dumps(index,indent=2)+'\n')
    (dst/'fp8-conversion.json').write_text(json.dumps({'source':str(src),'tensors':changed,'block_size':[128,128]},indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('destination');a=p.parse_args();convert(a.source,a.destination)
