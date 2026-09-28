"""dist48: default (a) vs low-rank-plane (lr) arm, vs same-H EXL3; keys routed / forced / OOD forced / OOD routed."""
import json, glob, sys
import numpy as np
OUT = "results_dist48"
KEYS = ("all/routed", "all/forced", "ood/forced", "ood/routed")
R = [json.load(open(f)) for f in sorted(glob.glob(f"{OUT}/L*_E*.json"))]
R = [r for r in R if "nq_lr/L4" in r["eval"]] if "--all" not in sys.argv else R
rel = lambda a, b: 100 * (a / b - 1)
rows = []
for r in R:
    e = r["eval"]; d = {}
    for arm, src in (("a", "nq"), ("lr", "nq_lr")):
        for Lv in (2, 4):
            for k in KEYS:
                d[(arm, Lv, k)] = rel(e[f"{src}/L{Lv}"][k], e[f"EXL3-{Lv}"][k])
    d["dbpw"] = r["rate"]["nq_lr"] - r["rate"]["nq"]
    d["rank"] = r["lr_meta"]["nq_lr"]["rank"]; d["nnz"] = r["lr_meta"]["nq_lr"]["nnz"]
    d["bitexact"] = all(all(v.values()) for v in r["lr_meta"]["nq_lr"]["bitexact"].values())
    d["enc"] = r["encode_s"]["nq_lr"]
    rows.append((r, d))
f = lambda d, arm, Lv: "/".join(f"{d[(arm, Lv, k)]:+6.1f}" for k in KEYS)
print(f"{'expert':<11}{'role':<13}| {'L2 a  r/F/oodF/oodR':<27}| {'L2 lr':<27}| {'L4 a':<27}| {'L4 lr':<27}| rank g/u/d  dbpw  enc")
for r, d in rows:
    rk = d["rank"]
    print(f"L{r['layer']:<2}E{r['expert']:<6}{r['role']:<13}| {f(d,'a',2)} | {f(d,'lr',2)} | {f(d,'a',4)} | {f(d,'lr',4)} | "
          f"{rk['gate']}/{rk['up']}/{rk['down']}{'s' if any(n and min(n) <= 8 for n in d['nnz'].values() if n) else ' '} {d['dbpw']:+.4f} {d['enc']:.0f}")
print(f"\nn = {len(rows)}; bit-exact all: {all(d['bitexact'] for _, d in rows)}; dbpw mean {np.mean([d['dbpw'] for _, d in rows]):+.4f} "
      f"max {max(d['dbpw'] for _, d in rows):+.4f}; lr encode s mean {np.mean([d['enc'] for _, d in rows]):.0f} max {max(d['enc'] for _, d in rows):.0f}")
for Lv in (2, 4):
    for arm in ("a", "lr"):
        for k in KEYS:
            x = np.array([d[(arm, Lv, k)] for _, d in rows])
            print(f"  L{Lv} {arm:<3} {k:<11} mean {x.mean():+6.2f} median {np.median(x):+6.2f} p90 {np.percentile(x, 90):+6.2f}"
                  f" worst {x.max():+7.2f}  #worse {int((x > 0).sum()):2d}  #>+1.5 {int((x > 1.5).sum()):2d}")
