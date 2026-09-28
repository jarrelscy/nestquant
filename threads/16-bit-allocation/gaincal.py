import sys; sys.path.insert(0,'/home/coder/git/nestquant/threads/05-exl3-harness')
import torch, harness as h; h.gpu_cap(12)
TQ = h.ExtTileQuantizer('mul1')
torch.manual_seed(0); x = torch.randn(2048, 256, device='cuda')
for K in [1, 1.5, 2, 2.5, 3, 4]:
    res = []
    for g in [0.6,0.7,0.8,0.85,0.9,0.95,1.0,1.05,1.1,1.2,1.3]:
        q,_ = TQ((x*g).contiguous(), K); res.append((float(((q.float()/g-x)**2).mean()), g))
    print(K, min(res), flush=True)
