"""Diagnostics on E36: seed noise, per-projection contribution, g-scale multiplier, two-sided G (+blend)."""
from core17 import *
X = Exp(16, 36)
M = {}
base = fit_expert(X)["W2"]; M["nat2"] = base
for s in [1, 2, 3]:
    M[f"seed{s}"] = fit_expert(X, seed=s)["W2"]
for p in range(3):
    M[f"only_{p}"] = [base[i] if i == p else X.W[i] for i in range(3)]
for gm in [0.9, 0.95, 1.05, 1.1]:
    M[f"gs{gm}"] = fit_expert(X, gscale_mult=gm)["W2"]
for b in [0.5, 1.0]:
    r = fit_expert(X, G_beta=b); M[f"G{b}"] = r["W2"]
r = fit_expert(X, G_beta=0.5, sig=SIG); 
for so in [0.1]:
    M[f"G0.5_so{so}"] = fit_expert(X, G_beta=0.5, sigma_out=so)["W2"]
r = fit_expert(X, G_beta=0.5, lam=0.3, level4=True); M["G0.5_b0.3@2"], M["G0.5_b0.3@4"] = r["W2"], r["W4"]
print("G+blend proxies", [(i["proxy2"], i["proxy4"]) for i in r["info"]])
res = X.capture(M)
for k, v in res.items(): print(f"{k:14s}", v, flush=True)
jdump(res, "results/diag_e36.json")
