"""How much does EXL3's 2-pass tail-biting heuristic lose vs better ring closure? (i.i.d. Gaussian, mul1 K=2)"""
from core17 import *
torch.manual_seed(1)
x = torch.randn(512, 256, device="cuda")
q, _ = EXT(x, 2); print("CUDA 2-pass", (q - x).square().mean().item())
TQ = h.TorchTileQuantizer(h.codebook_lut("mul1"))
qt, it = TQ(x, 2); print("torch 2-pass", (qt - x).square().mean().item())
# open (non-tail-biting) trellis lower bound: min over all paths without ring constraint
lut = TQ.lut; K = 2; Kr = 14; E = 1 << Kr
e = torch.arange(E, device="cuda"); k = torch.arange(4, device="cuda")
states = (k[:, None] << Kr) | e[None, :]; prev = states >> K; vals = lut[states]
def open_cost(w, roll=0):
    cost = None
    for i in range(256):
        d = (vals[None] - w[:, (i + roll) % 256, None, None]).square()
        c = d if cost is None else d + cost[:, prev]
        cost = c.min(1).values
    return cost.min(1).values
lb = open_cost(x); print("open-trellis lower bound", (lb.sum() / x.numel()).item())
# multi-start: constrained pass 2 for several candidate start edges from different rolls
best = None
for roll in [128, 64, 192, 32]:
    TQ2 = h.TorchTileQuantizer(h.codebook_lut("mul1"))
    xr = x.roll(-roll + 128, 1)
    q2, _ = TQ2(xr, 2); q2 = q2.roll(roll - 128, 1)
    m = (q2 - x).square().sum(1)
    best = m if best is None else torch.minimum(best, m)
    print("multi-roll up to", roll, (best.sum() / x.numel()).item(), flush=True)
