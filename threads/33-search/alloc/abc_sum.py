#!/usr/bin/env python3
"""summarise abc json(s) at matched upload bytes per token: churn per refresh target = C16 * R / 16.
  abc_sum.py FILE [C16 ...]  (default 2.2 2.8 3.2); optional env W=kld for the KLD-weighted layer mean."""
import sys, json, os
import numpy as np
f = sys.argv[1]; C = [float(x) for x in sys.argv[2:]] or [2.2, 2.8, 3.2]
d = json.load(open(f)); R = d["R"]; pl = d["per_layer"]
kw = json.load(open("/tmp/nestquant/33-search/alloc/kldw.json")) if os.environ.get("W") == "kld" else None
Ls = list(pl)
wt = np.array([kw[L] for L in Ls]) if kw else np.ones(len(Ls)); wt = wt / wt.sum()
keys = pl[Ls[0]].keys()
nfs = sorted({k.split("|")[0] for k in keys}, key=lambda x: (len(x), x))
MB = 75 * 10.24 / R          # MB per token per unit churn/refresh
res = {}
for nf in nfs:
    hms = sorted(float(k.split("|")[1]) for k in keys if k.split("|")[0] == nf)
    s = np.array([sum(wt[i] * pl[L][f"{nf}|{h}"]["sal"] for i, L in enumerate(Ls)) for h in hms]) * 100
    c = np.array([np.mean([pl[L][f"{nf}|{h}"]["churn"] for L in Ls]) for h in hms])
    o = np.argsort(c)
    vals = []
    for c16 in C:
        tc = c16 * R / 16
        v = np.interp(tc, c[o], s[o], left=np.nan, right=np.nan)
        vals.append(v)
    res[nf] = vals
    vr = "" if nf == "a" else f"  VRAM {(int(nf) - 77) * 0.768:+6.2f} GB"
    print(f"R{R:<2d} nf {nf:>4s}  " + "  ".join(f"@{c16}/16tok (churn {c16 * R / 16:.2f}, {c16 * MB * R / 16 / 1e0:.0f} MB/tok) {v:6.2f}"
                                          for c16, v in zip(C, vals)) + vr)
json.dump(res, open(f.replace(".json", "_sum.json"), "w"))
