# J4 fitted jointly (product-state Viterbi): L2 = mul1(b16), L4 = mul1(8b base | 4b P3 | 4b P4); cost a*D2 + b*D4
import torch, math, json
import joint as J
from trellis import mul1_codebook
torch.manual_seed(0)
N = 8
x = torch.randn(N, 256, device='cuda')
db = lambda m, R: 10 * math.log10(m / 2 ** (-2 * R))
M = mul1_codebook(16)
Wi = torch.arange(2 ** 24, device='cuda')
r8 = Wi & 255; b8 = (Wi >> 8) & 255
p3 = sum((((r8 >> (2 * j + 1)) & 1) << j) for j in range(4)); p4 = sum((((r8 >> (2 * j)) & 1) << j) for j in range(4))
G4 = M[(b8 << 8) | (p3 << 4) | p4]
F2full = M[Wi >> 8]
G = G4 - F2full            # joint.py models L4 = f2 + G
res = {}
for s in [0.95]:
    for a, b in [(1, 1), (1, 4), (0, 1)]:
        Ws = torch.cat([J.joint_viterbi(x[i:i + 4] * s, G, a, b) for i in range(0, N, 4)])
        D2 = ((x - M[Ws >> 8] / s) ** 2).mean().item(); D4 = ((x - G4[Ws] / s) ** 2).mean().item()
        print(f"J4 joint a={a} b={b}: D2={D2:.5f} ({db(D2,2):+.2f} dB) D4={D4:.6f} ({db(D4,4):+.2f} dB)", flush=True)
        res[f"{a}_{b}"] = (D2, D4)
json.dump(res, open('exp5_joint.json', 'w'), indent=1)
