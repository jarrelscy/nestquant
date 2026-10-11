"""Batched independent jMT residual heads; unchanged projection and softmax."""
import torch
from torch import nn
from torch.nn import functional as F

class BatchedHeads(nn.Module):
 def __init__(self, projection, down, up, bias):
  super().__init__()
  self.projection=projection
  self.bias=bias
  self.register_buffer('down_weight',torch.stack([m.weight.detach() for m in down]),persistent=False)
  self.register_buffer('up_weight',torch.stack([m.weight.detach() for m in up]),persistent=False)
 def forward(self,x,first,last):
  # x: [B,T,D]; output: [H,B,T,E]. No mixing of horizon logits.
  d=self.down_weight[first-1:last];u=self.up_weight[first-1:last]
  y=F.gelu(torch.einsum('btd,hrd->hbtr',x,d))
  z=x[None]+torch.einsum('hbtr,hdr->hbtd',y,u)
  return self.projection(z)+self.bias[first-1:last,None,None,:]
