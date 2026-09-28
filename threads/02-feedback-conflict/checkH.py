from common import *
import torch.nn.functional as F
Ws, Hs = glm()
ts = torch.load('/home/coder/git/orbit-duet/runs/glm53_pilot_matched_l16/statistics/l16_e36_training_sample.pt', map_location='cpu', mmap=True, weights_only=False)
G0 = torch.zeros(6144,6144,device='cuda'); G1 = torch.zeros(2048,2048,device='cuda')
g,u,d = Ws
for a in range(0, len(ts['x']), 2048):
    x = ts['x'][a:a+2048].cuda().float(); p = ts['p'][a:a+2048].cuda()[:,None]
    hid = (F.silu(F.linear(x.bfloat16(), g.bfloat16())) * F.linear(x.bfloat16(), u.bfloat16())).float()
    G0.addmm_((x*p).T, x*p); G1.addmm_((hid*p).T, hid*p)
print(float((G0-Hs[0]).norm()/Hs[0].norm()), float((G1-Hs[2]).norm()/Hs[2].norm()))
print(ts['p'][:5], ts['p'][-5:], ts['positions'][:5])
