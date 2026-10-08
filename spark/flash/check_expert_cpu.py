"""Compare packed loader decode with the checkpoint reference, without a GPU."""
import argparse,json,os,sys,types
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('--layer',type=int,default=3)
p.add_argument('--repack');p.add_argument('--expert',type=int,default=0);p.add_argument('--tp',type=int,default=8);a=p.parse_args()
r=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(r/'sm120'),str(r/'streaming')]
build=types.ModuleType('build');build.get=lambda:None;sys.modules['build']=build
import nqload,moe
parts={i:nqload.load_part(nqload.layer_dir(a.checkpoint,a.layer),i) for i in range(8//a.tp)}
man=json.load(open(Path(nqload.layer_dir(a.checkpoint,a.layer))/'manifest.json'))
art=nqload.group_art(parts,man,a.expert,list(parts))
ex,scales,Q=nqload.kernel_expert(art,dev='cpu',want_Q=True)
result=[]
for projection,P,ref in [('gate_up',ex.gu,{lv:torch.cat([Q[0][lv],Q[1][lv]]) for lv in (2,4)}),('down',ex.dn,Q[2])]:
 for lv in (2,4):
  got=moe.dense_W(P,lv,G=4)
  expected=ref[lv].float()
  diff=(got-expected).abs()
  result.append(dict(projection=projection,level=lv,max_abs=diff.max().item(),relative_l2=(diff.norm()/expected.norm()).item(),equal=torch.equal(got,expected)))
if a.repack:
 import resident,p4rec
 rp=Path(a.repack);idx=json.load(open(rp/'rank0.json'))
 loaded,H,I=resident.load(str(rp/f'res/rank0/L{a.layer}.pt'),dev='cpu');other=loaded[a.expert]
 for proj in ('gu','dn'):
  for field in ('base','var'):
   assert torch.equal(getattr(getattr(ex,proj),field),getattr(getattr(other,proj),field)),(proj,field)
 for lv in (2,4):assert torch.equal(ex.sc[lv],other.sc[lv]),('sc',lv)
 if ex.lr is not None:assert torch.equal(ex.lr,other.lr),'lr'
 lay=dict(seg=idx['seg'],rec_bytes=idx['rec_bytes'])
 offset=((a.layer-idx['L0'])*idx['NE']+a.expert)*idx['rec_bytes']
 with (rp/'rank0.bin').open('rb') as f:
  f.seek(offset);actual=f.read(idx['rec_bytes'])
 assert actual==p4rec.pack(ex,lay),'residual record bytes'
 result.append(dict(repack_record_equal=True,resident_equal=True,layer=a.layer,expert=a.expert))
print(json.dumps(result,indent=2),flush=True)
if not all(x.get('equal',True) for x in result):raise SystemExit('Packed decode differs from checkpoint reference')
