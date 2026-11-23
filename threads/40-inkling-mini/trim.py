"""Trimmed KLD: drop tokens where the native-weight floor run exceeds 0.1 nats (eval-path knife-edge tokens)."""
import sys, torch
def load(d, W):
    out = {}
    for r in range(W):
        for si, k in torch.load(f"{d}/final/klv.r{r}.pt"):
            out[(r, si)] = k
    return out
pre, W = sys.argv[1], int(sys.argv[2])
nat = load(f"/tmp/nq_mini/kld/{pre}_native", W)
keys = sorted(nat)
mask = torch.cat([nat[k] <= 0.1 for k in keys])
print(f"{pre}: knife-edge tokens {int((~mask).sum())} / {len(mask)}")
for tag in ["native", "2", "4", "hot56", "float"]:
    try:
        v = load(f"/tmp/nq_mini/kld/{pre}_{tag}", W)
    except FileNotFoundError:
        continue
    x = torch.cat([v[k] for k in keys])
    print(f"  {tag:7s} mean {float(x.mean()):.4f}  trimmed {float(x[mask].mean()):.4f}  median {float(x.median()):.5f}")
