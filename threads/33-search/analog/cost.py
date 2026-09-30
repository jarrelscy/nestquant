"""serve cost of the analog query per 16-token step for 75 layers (CPU torch, 1 thread and 8 threads; GPU if free is
not used).  state update (3 EMAs x 256, sqrt, PCA 768x64) + kNN top-32 over a bank of N x 64 + gather-mean of 3x256."""
import sys
import time
import torch
for thr in (1, 8):
    torch.set_num_threads(thr)
    for N in (4096, 16384, 65536):
        K = torch.randn(75, N, 64); kn = (K ** 2).sum(-1); V = torch.randn(75, N, 768, dtype=torch.float16)
        W = torch.randn(75, 768, 64); E = torch.rand(75, 3, 256)
        s = torch.rand(75, 256)
        def step():
            E.mul_(0.9).add_(s[:, None])
            z = (E / E.sum(-1, keepdim=True)).sqrt().reshape(75, 1, 768)
            q = torch.bmm(z, W)                                 # [75,1,64]
            d2 = kn[:, None] - 2 * torch.bmm(q, K.transpose(1, 2))
            ix = torch.topk(d2, 32, -1, largest=False)[1][:, 0]  # [75,32]
            return torch.stack([V[l, ix[l]].float().mean(0) for l in range(75)])
        for _ in range(3):
            step()
        t = time.time(); n = 20
        for _ in range(n):
            step()
        ms = (time.time() - t) / n * 1e3
        mb = 75 * N * (64 * 4 + 768 * 2) / 2 ** 20
        print(f"threads {thr} bank/layer {N:6d}: {ms:7.2f} ms/step (75 layers)  bank mem {mb:7.0f} MB (fp32 keys, fp16 3x256 values)",
              flush=True)
