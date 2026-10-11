import pathlib,sys,unittest
import torch
from torch import nn
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'spark/flash'))
from jmt_heads import BatchedHeads
class TestHeads(unittest.TestCase):
 def test_independent_heads_and_chunk_sizes(self):
  torch.manual_seed(4)
  p=nn.Linear(24,96);down=nn.ModuleList([nn.Linear(24,8,bias=False) for _ in range(16)])
  up=nn.ModuleList([nn.Linear(8,24,bias=False) for _ in range(16)])
  bias=nn.Parameter(torch.randn(16,96));batched=BatchedHeads(p,down,up,bias)
  for n in (1,3,128):
   x=torch.randn(1,n,24)
   for first,last in ((1,16),(3,5),(16,16)):
    ref=torch.stack([p(x+up[h](torch.nn.functional.gelu(down[h](x))))+bias[h] for h in range(first-1,last)])
    torch.testing.assert_close(batched(x,first,last),ref,atol=2e-6,rtol=2e-5)
