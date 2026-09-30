#!/usr/bin/env python3
"""Per-projection 2->4 error-energy drop from the served per-layer manifests (HF layers/L{L}/manifest.json,
proxy_rot[v] per gate/up/down) + T31's G -> $A/proj_delta.npz: dp [75,256,3] (gate, up, down) = p2 - p4,
delta [75,256] (T31), G [75,256].  Check: T31 delta == mean_proj(dp) * G."""
import json
import numpy as np
A = "/tmp/nestquant/33-search/alloc"
t = np.load("/tmp/nestquant/31-delta/delta_table.npz")
LAYERS = list(t["layers"])
dp = np.zeros((75, 256, 3)); p2 = np.zeros((75, 256, 3)); p4 = np.zeros((75, 256, 3))
bits = {}
for i, L in enumerate(LAYERS):
    m = json.load(open(f"{A}/hfman/layers/L{L}/manifest.json"))
    for e in range(256):
        pe = m["per_expert"][str(e)]["proj"]
        for j, pn in enumerate(("gate", "up", "down")):
            p2[i, e, j] = pe[pn]["proxy_rot"]["2"]; p4[i, e, j] = pe[pn]["proxy_rot"]["4"]
    bits[L] = {pn: m["per_expert"]["0"]["proj"][pn]["bits"] for pn in ("gate", "up", "down")}
dp = p2 - p4
d_chk = dp.mean(2) * t["G"]
rel = np.abs(d_chk / t["delta"] - 1)
print("T31 delta reproduced: max rel err", rel.max(), " (L3-6 max", rel[:4].max(), ")")
share = dp / dp.sum(2, keepdims=True)
print("down share of drel (kappa=1): mean %.3f  p10 %.3f p90 %.3f" % (share[..., 2].mean(), *np.percentile(share[..., 2], [10, 90])))
for b, sl in (("L3-6", slice(0, 4)), ("L7-40", slice(4, 38)), ("L41-77", slice(38, 75))):
    print(b, "down share %.3f  p2 g/u/d %s  p4 %s" % (share[sl, :, 2].mean(), p2[sl].mean((0, 1)).round(4), p4[sl].mean((0, 1)).round(4)))
print("bits L10", bits[10])
np.savez(f"{A}/proj_delta.npz", dp=dp, p2=p2, p4=p4, delta=t["delta"], G=t["G"], layers=np.array(LAYERS))
