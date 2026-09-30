#!/usr/bin/env python3
"""interpolate each k's (churn, metric) hm-curve at target churns (linear in churn between neighbouring hm)."""
import json, sys
import numpy as np
f = sys.argv[1]
tg = [float(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "2.8,3.2").split(",")]
S = json.load(open(f))["summary"]
ks = sorted({int(k.split("|")[0]) for k in S})
for k in ks:
    rows = sorted([(v["churn"], v) for key, v in S.items() if int(key.split("|")[0]) == k], key=lambda r: r[0])
    c = np.array([r[0] for r in rows])
    out = []
    for t in tg:
        g = {m: float(np.interp(t, c, [r[1][m] for r in rows])) for m in ("sal", "cnt", "sal_early", "sal_late")}
        hm = float(np.interp(t, c, [float(key.split("|")[1]) for key, v in sorted(
            [(key, v) for key, v in S.items() if int(key.split("|")[0]) == k], key=lambda kv: kv[1]["churn"])]))
        out.append(f"@{t}: sal {g['sal']*100:6.2f} routes {g['cnt']*100:6.2f} early256 {g['sal_early']*100:6.2f} (hm~{hm:.2f})")
    print(f"k{k:3d}  " + " | ".join(out))
