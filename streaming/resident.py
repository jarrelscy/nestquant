"""Resident (level-2) planes of one TP rank, in the serving format (nq-res-v1): one file per (rank, layer) holding
everything the GPU keeps for the whole run: 2-bit base + base-variant planes, level-2 and level-4 scale vectors, the
low-rank plane (V, U2), the residual K codes and (threads/29, only when != 128) the layer's in_had_down. The P4 part of every expert is in the record file (repack.py
rank{r}.bin); fixed-set experts read their record once at startup into a resident pool, the rest stream.
  save(RL, path)                 nqload.RankLayer -> path (stacked [E, ...] tensors)
  load(path, dev) -> (ex{E}, H, I)  kernel experts with the fields moe.entry / p4rec.row read (level 2 rows; the level-4
                                 row points the P4 fields into a slot)
RL = nqload.RankLayer; ex.lr4 of a loaded expert is a dummy (the kernel reads U4 from the slot at level 4)."""
import types,torch
RMAX=4

def lr_len(H,I,rg,rd):return rg*(H+2*I)+rd*(I+H)

def save(RL,path):
    E=RL.experts;ex=[RL.ex[e] for e in E];H,I=RL.H,RL.I;n=lr_len(H,I,RMAX,RMAX)
    st=lambda f:torch.stack([f(x).cpu() for x in ex])
    lr=torch.zeros(len(E),n,dtype=torch.float16)
    for i,x in enumerate(ex):
        if x.lr is not None:lr[i,:x.lr.numel()]=x.lr.cpu()
    d=dict(format='nq-res-v1',L=RL.L,rank=RL.rank,tp=RL.tp,H=H,I=I,experts=list(E),
           rk_gu=[x.gu.rk for x in ex],rk_dn=[x.dn.rk for x in ex],rg=[x.rg for x in ex],rd=[x.rd for x in ex],
           gu_base=st(lambda x:x.gu.base),gu_var=st(lambda x:x.gu.var),dn_base=st(lambda x:x.dn.base),dn_var=st(lambda x:x.dn.var),
           sc2=st(lambda x:x.sc[2]),sc4=st(lambda x:x.sc[4]),lr=lr)
    w=int(getattr(RL,'had_dn',128))
    if w!=128:d['in_had_down']=w                          # threads/29; absent = 128 (older files stay byte-identical)
    torch.save(d,path+'.tmp');import os;os.replace(path+'.tmp',path)

def load(path,dev='cuda'):
    d=torch.load(path,map_location='cpu',mmap=True,weights_only=False);assert d['format']=='nq-res-v1',d.get('format')
    H,I=d['H'],d['I'];g={k:d[k].to(dev) for k in ('gu_base','gu_var','dn_base','dn_var','sc2','sc4','lr')}
    dummy=torch.zeros(1,dtype=torch.float16,device=dev);out={}
    hw=had_width(path,d)
    for i,E in enumerate(d['experts']):
        P=lambda k,rk:types.SimpleNamespace(base=g[k+'_base'][i],var=g[k+'_var'][i],p4=None,d4=None,flags=None,fl=None,rk=rk)
        rg,rd=d['rg'][i],d['rd'][i];has=rg+rd>0
        x=types.SimpleNamespace(gu=P('gu',d['rk_gu'][i]),dn=P('dn',d['rk_dn'][i]),H=H,I=I,rg=rg,rd=rd,
                                sc={2:g['sc2'][i],4:g['sc4'][i]},lr=g['lr'][i,:lr_len(H,I,rg,rd)] if has else None,lr4=dummy if has else None)
        x.signs=x.sc[2];x.had_dn=hw;out[E]=x
    return out,H,I

def had_width(path,d):
    """threads/29 in_had_down of a res file: the .pt's own key when present (our repack.py), else the layer entry of
    <repack>/rank{r}.json (the HF serving/tp4 copy keeps it only there), absent = 128"""
    if 'in_had_down' in d:return int(d['in_had_down'])
    import os,json
    ip=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(path)))),f"rank{d['rank']}.json")
    return int(_rank_layers(ip).get(str(d['L']),{}).get('in_had_down',128))

import functools
@functools.lru_cache(None)
def _rank_layers(ip):
    import os,json
    return json.load(open(ip))['layers'] if os.path.exists(ip) else {}
