"""MTP routed experts BF16 -> e4m3 with 128x128 block scales (DeepSeek-V3 fp8 layout: <name>.weight e4m3 +
<name>.weight_scale_inv f32 [ceil(N/128), ceil(K/128)], dequant = w * scale). Same math as vLLM's
per_block_cast_to_fp8 (use_ue8m0=False). The backbone checkpoint keeps the MTP layer's 256 experts in BF16 (19.3 GB,
9 GiB per node at TP2); this halves them. Served by spark/overlay/nvfp4_arvq_hybrid.py with NQ_MTP_FP8=1.

  python mtp_fp8.py SRC DST
SRC holds the shards that contain the MTP experts (start_spark.sh downloads them there); every tensor of those shards
goes into DST/nq-mtp-fp8.safetensors, MTP routed-expert weights converted, the rest copied. DST/model.safetensors.index.json
is repointed at it. The conversion is skipped when the output exists; the index patch always runs (hf download restores
the original index whenever it re-checks the repo)."""
import os,re,sys,json,torch
from safetensors import safe_open
from safetensors.torch import save_file
OUT='nq-mtp-fp8.safetensors';B=128

def q_block(w):
    m,n=w.shape;xp=torch.zeros((m+B-1)//B*B,(n+B-1)//B*B,dtype=w.dtype);xp[:m,:n]=w
    xv=xp.view(-1,B,xp.size(1)//B,B);amax=xv.abs().float().amax(dim=(1,3),keepdim=True).clamp(1e-4)
    sf=amax/448.0;q=(xv*(1.0/sf)).to(torch.float8_e4m3fn)
    return q.view_as(xp)[:m,:n].contiguous(),sf.view(xv.size(0),xv.size(2)).contiguous()

def mtp_layer(wm):
    return max(int(m.group(1)) for n in wm for m in [re.search(r'layers\.(\d+)\.mlp\.experts\.\d+\.',n)] if m)

def is_mtp_expert(n,L):return re.search(rf'(^|\.)layers\.{L}\.mlp\.experts\.\d+\.(gate|up|down)_proj\.weight$',n) is not None

def main(src,dst):
    ip=os.path.join(dst,'model.safetensors.index.json');idx=json.load(open(ip));wm=idx['weight_map']
    L=mtp_layer(wm);shards=sorted({f for n,f in wm.items() if is_mtp_expert(n,L)}-{OUT})
    out=os.path.join(dst,OUT)
    if not os.path.exists(out):
        assert shards,'no MTP expert shards in the index'
        T={};nq=0;torch.set_num_threads(min(16,os.cpu_count() or 1))
        for s in shards:
            with safe_open(os.path.join(src,s),'pt') as f:
                for n in f.keys():
                    t=f.get_tensor(n)
                    if is_mtp_expert(n,L):
                        T[n],T[n+'_scale_inv']=q_block(t);nq+=1
                    else:T[n]=t
            print(f'  {s}: done ({nq} expert weights converted so far)',flush=True)
        save_file(T,out+'.tmp',metadata={'format':'pt'});os.replace(out+'.tmp',out)
        print(f'wrote {out}: {nq} MTP expert weights (layer {L}) in e4m3 + 128x128 scales, {os.path.getsize(out)/2**30:.2f} GiB')
    with safe_open(out,'pt') as f:names=set(f.keys())
    for n in [n for n,f in wm.items() if f in shards]:del wm[n]
    for n in names:wm[n]=OUT
    tmp=ip+'.tmp';json.dump(idx,open(tmp,'w'),indent=2);os.replace(tmp,ip)
    print(f'index: {len(names)} tensors -> {OUT} (replaces {", ".join(shards) or "nothing, already patched"})')

if __name__=='__main__':main(*sys.argv[1:3])
