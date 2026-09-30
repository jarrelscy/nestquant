#!/usr/bin/env python3
"""pooled sal-hot (sum over layers of salience on 4-bit / sum of total) from abc.py JSONs, beside the layer mean.
Layer totals = corpus bsal sums from the v2 cache (R-invariant).   pooled.py FILE [C16 ...]"""
import sys, json
import numpy as np
sys.path.insert(0, "/home/coder/git/nestquant/threads/33-search/alloc")
f = sys.argv[1]; C = [float(x) for x in sys.argv[2:]] or [2.2, 2.8]
d = json.load(open(f)); R = d["R"]; pl = d["per_layer"]; corpus = d["corpus"]
Ls = list(pl)
tot = np.array([np.load(f"/tmp/nestquant/33-search/alloc/cache/{corpus}/L{L}.npz")["bsal"].astype(np.float64).sum() for L in Ls])
wp = tot / tot.sum()
keys = pl[Ls[0]].keys(); nfs = sorted({k.split("|")[0] for k in keys}, key=lambda x: (len(x), x))
for nf in nfs:
    hms = sorted(float(k.split("|")[1]) for k in keys if k.split("|")[0] == nf)
    c = np.array([np.mean([pl[L][f"{nf}|{h}"]["churn"] for L in Ls]) for h in hms]); o = np.argsort(c)
    row = []
    for nm, w in (("pooled", wp), ("mean", np.ones(len(Ls)) / len(Ls))):
        s = np.array([sum(w[i] * pl[L][f"{nf}|{h}"]["sal"] for i, L in enumerate(Ls)) for h in hms]) * 100
        row += [f"{nm} " + " ".join(f"{np.interp(t * R / 16, c[o], s[o], left=np.nan, right=np.nan):6.2f}" for t in C)]
    print(f"{corpus:14s} R{R:<2d} {nf:>4s}  " + "  |  ".join(row))
