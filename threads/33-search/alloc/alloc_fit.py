#!/usr/bin/env python3
"""T33k alloc (B): per-layer slot allocation at equal total VRAM (sum nf = TOT), greedy on each layer's marginal
sal-hot gain per slot (concave envelope of the calib-fit curve at hm HM, linear interpolation between grid points).
  alloc_fit.py CURVES.json HM TOT OUT.json [kld]   (kld: gains weighted by T18 per-layer KLD-increment weight)"""
import sys, json
import numpy as np

cur = json.load(open(sys.argv[1]))["per_layer"]; HM = sys.argv[2]; TOT = int(sys.argv[3]); out = sys.argv[4]
kw = json.load(open(sys.argv[5] if sys.argv[5] != "kld" else "/tmp/nestquant/33-search/alloc/kldw.json")) if len(sys.argv) > 5 else None
alloc, curves = {}, {}
for L, d in cur.items():
    nfs = sorted({int(k.split("|")[0]) for k in d})
    nfs = [n for n in nfs if n >= int(__import__("os").environ.get("FLOOR", "0"))]
    y = np.array([d[f"{n}|{HM}"]["sal"] for n in nfs])
    xs = np.arange(nfs[0], nfs[-1] + 1)
    yi = np.interp(xs, nfs, y)
    # concave upper envelope
    hull = [0]
    for j in range(1, len(xs)):
        hull.append(j)
        while len(hull) >= 3:
            a, b, c = hull[-3:]
            if (yi[b] - yi[a]) * (xs[c] - xs[a]) <= (yi[c] - yi[a]) * (xs[b] - xs[a]):
                hull.pop(-2)
            else:
                break
    ye = np.interp(xs, xs[hull], yi[hull])
    wt = (kw[L] + (1e-4 if sys.argv[5] == "kld" else 0)) if kw else 1.0
    curves[L] = (xs, ye * wt)
    alloc[L] = int(xs[0])
left = TOT - sum(alloc.values())
assert left >= 0
import heapq
h = []
for L, (xs, ye) in curves.items():
    h.append((-(ye[1] - ye[0]), L))
heapq.heapify(h)
while left > 0:
    g, L = heapq.heappop(h)
    alloc[L] += 1; left -= 1
    xs, ye = curves[L]; j = alloc[L] - xs[0]
    if j + 1 < len(xs):
        heapq.heappush(h, (-(ye[j + 1] - ye[j]), L))
json.dump(alloc, open(out, "w"))
v = np.array([alloc[str(L)] for L in range(3, 78)])
print("alloc", v.tolist(), "sum", v.sum(), "min", v.min(), "max", v.max())
print("bands L3-6 %.1f L7-40 %.1f L41-77 %.1f" % (v[:4].mean(), v[4:38].mean(), v[38:].mean()))
