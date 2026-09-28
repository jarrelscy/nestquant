"""Headroom diagnostic (NOT a candidate): fit with a Hessian mixed with the evaluation capture's own statistics.
Tells how much of the 2-bit capture error is generalization (objective) vs coding."""
from core17 import *
X = Exp(16, 36)
cap = X.data.capture
g, u, d = X.W
routed, slots = torch.where(cap["ids"] == 36)
pr = torch.zeros(len(cap["x"])); pr[routed] = cap["p"][routed, slots]
Hx = torch.zeros(6144, 6144, device="cuda", dtype=torch.float64); Ha = torch.zeros(2048, 2048, device="cuda", dtype=torch.float64)
Hxr = torch.zeros_like(Hx); Har = torch.zeros_like(Ha)
for i in range(0, len(cap["x"]), 512):
    x = cap["x"][i:i+512].cuda().float(); w = pr[i:i+512].cuda()[:, None]
    a = (F.silu(F.linear(x.bfloat16(), g.bfloat16())) * F.linear(x.bfloat16(), u.bfloat16())).float()
    Hx.addmm_(x.double().T, x.double()); Ha.addmm_(a.double().T, a.double())
    xr, ar = (x * w).double(), (a * w).double(); Hxr.addmm_(xr.T, xr); Har.addmm_(ar.T, ar)
nt = lambda A: (A / A.diagonal().mean()).float()
Hcap = [0.5 * nt(Hx) + 0.5 * nt(Hxr), 0.5 * nt(Hx) + 0.5 * nt(Hxr), 0.5 * nt(Ha) + 0.5 * nt(Har)]
M = {"nat2": fit_expert(X)["W2"]}
for mix in [0.5, 1.0]:
    for sg in [(0.5, 0.5, 1.0), (0.1, 0.1, 0.1)]:
        W2 = []
        for p in range(3):
            H = (1 - mix) * X.H(p) + mix * Hcap[p]
            W2.append(quantize(X.W[p], H, sigma=sg[p])["W2"])
        M[f"oracle_mix{mix}_s{sg[0]}"] = W2
res = X.capture(M)
for k, v in res.items(): print(f"{k:24s}", v, flush=True)
jdump(res, "results/oracle_e36.json")
