import sys, time, torch
sys.path.insert(0,'/home/coder/git/nestquant/threads/05-exl3-harness')
import harness as h; h.gpu_cap(12)
Q = h.ExtTileQuantizer('mul1')
x = torch.randn(8, 256, device='cuda')
for K in [2,4]:
    best=None
    for g in [0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4]:
        xs = torch.randn(512,256,device='cuda')
        q,_ = Q(xs*g, K); m = float(((q/g)-xs).square().mean())
        print(K, g, round(m,5))
    torch.cuda.synchronize(); t=time.time()
    for _ in range(100): Q(x, K)
    torch.cuda.synchronize(); print('ms/call 8 tiles', (time.time()-t)*10)
