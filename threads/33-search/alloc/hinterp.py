#!/usr/bin/env python3
"""hedge frontier: for each (k, nF, nD), the upper envelope of metric vs byte-churn over (hm_any, hm_full), read at
target churns (linear interpolation along the envelope)."""
import json, sys
import numpy as np
f = sys.argv[1]; tg = [float(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "2.8,3.2").split(",")]
mets = sys.argv[3].split(",") if len(sys.argv) > 3 else ["sal_k0.7", "sal_k0.9", "sal_k1.0", "sal_k1.4", "dsal_k1.0", "full", "any"]
S = json.load(open(f))["summary"]
groups = {}
for key, v in S.items():
    k, nF, nD, ha, hf = key.split("|")
    groups.setdefault((int(k), int(nF), int(nD)), []).append(v)


def env(pts, m, t):
    """max over configs of metric at churn <= t, with linear interpolation between envelope points."""
    P = sorted((p["churn_b"], p[m]) for p in pts)
    best = []
    for c, y in P:                                  # upper-left envelope (monotone in churn)
        if not best or y > best[-1][1]:
            best.append((c, y))
    c = np.array([b[0] for b in best]); y = np.array([b[1] for b in best])
    if t < c[0]:
        return float("nan")
    return float(np.interp(t, c, y))


for g in sorted(groups):
    pts = groups[g]
    if not all(m in pts[0] for m in mets):
        mm = [m for m in mets if m in pts[0]]
    else:
        mm = mets
    print(f"k{g[0]:2d} nF{g[1]:3d} nD{g[2]:3d}  " + " | ".join(
        f"@{t}: " + " ".join(f"{m.replace('sal_', '')} {env(pts, m, t) * 100:6.2f}" for m in mm) for t in tg))
