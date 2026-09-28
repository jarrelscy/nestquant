"""Reproduce anchors on E36: EXL3-2/4 (harness) and my generic loop (nat2, blend0.3 L2/L4)."""
from core17 import *
X = Exp(16, 36)
t = T0()
M = {}
M["exl3_2"] = exl3_anchor(X, 2); print("exl3_2", time.time() - t, flush=True)
M["exl3_4"] = exl3_anchor(X, 4); print("exl3_4", time.time() - t, flush=True)
r = fit_expert(X, level4=True, lam=0.0); M["seq@2"], M["seq@4"] = r["W2"], r["W4"]; print("seq", [i["proxy2"] for i in r["info"]], [i["proxy4"] for i in r["info"]], time.time() - t, flush=True)
r = fit_expert(X, level4=True, lam=0.3); M["b0.3@2"], M["b0.3@4"] = r["W2"], r["W4"]; print("b03", time.time() - t, flush=True)
res = X.capture(M)
for k, v in res.items(): print(f"{k:10s}", v)
jdump(res, "results/repro_e36.json")
