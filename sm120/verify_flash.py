"""SM120 Flash checks, run only on a leased idle GPU.
Checks packed decode (decode and prefill paths) bitwise against dense_W, then
full expert outputs with Flash's clamp at levels 2/4 and B=1,2,4,8.
"""
import os
os.environ.setdefault('NQ_DEFS','NQ_WDUMP,NQ_RK_CODES=0x405,NQ_RK_GU=0x5,NQ_RK_DN=0x401,NQ_BK_CODES=0x21,NQ_SWIGLU_LIMIT=10')
import json,torch
import moe
from moe import Expert,MoELayer,dense_W

torch.manual_seed(717)
H,I=4096,2048
ex=Expert(H,I,seed=1337,rk_gu=2,rk_dn=10,bk_gu=5,bk_dn=5,var=True)
M=moe.M
assert M.swiglu_limit()==10
layer=MoELayer(1,H,I,Bmax=8,G=4)
layer.cfg_gu=layer.cfg_dn=[1,8,2]
rows=[]
for level in (2,4):
 layer.set(0,ex,level)
 wg=torch.empty(2*I,H,device='cuda',dtype=torch.float16)
 wd=torch.empty(H,I,device='cuda',dtype=torch.float16)
 pg=torch.empty_like(wg);pd=torch.empty_like(wd)
 for batch in (1,2,4,8):
  x=(torch.randn(batch,H,device='cuda')*.05).half()
  ids=torch.zeros(batch,8,dtype=torch.long,device='cuda')
  weights=torch.zeros(batch,8,dtype=torch.float16,device='cuda');weights[:,0]=1
  M.set_wdump(wg.data_ptr(),wd.data_ptr(),0)
  got=layer(x,ids,weights).clone();torch.cuda.synchronize();M.set_wdump(0,0,-1)
  for name,p,decoded in [('gu',ex.gu,wg),('down',ex.dn,wd)]:
   expected=dense_W(p,level,4,torch.float16)
   if not torch.equal(decoded,expected):raise AssertionError(f'decode {name} level={level} B={batch}')
  expected=ex.ref(x.float(),level)
  delta=got-expected
  rel=(delta.norm()/expected.norm().clamp_min(1e-20)).item()
  rows.append(dict(path='decode',level=level,batch=batch,max_abs=delta.abs().max().item(),relative_l2=rel))
  if not torch.isfinite(got).all() or rel>.01:raise AssertionError(rows[-1])
 M.pf_decode(layer.table,torch.tensor([0],dtype=torch.int32,device='cuda'),pg,pd,H,I,0,0)
 for name,p,decoded in [('gu',ex.gu,pg),('down',ex.dn,pd)]:
  if not torch.equal(decoded,dense_W(p,level,4,torch.float16)):raise AssertionError(f'prefill {name} level={level}')
 got=layer.prefill(x,ids,weights)
 delta=got-expected;rel=(delta.norm()/expected.norm().clamp_min(1e-20)).item()
 rows.append(dict(path='prefill',level=level,batch=batch,max_abs=delta.abs().max().item(),relative_l2=rel))
 if not torch.isfinite(got).all() or rel>.01:raise AssertionError(rows[-1])
print(json.dumps(rows,indent=2))
