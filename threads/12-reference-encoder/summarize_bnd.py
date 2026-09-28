"""Boundary damping A/B summary: per arm, % change vs the w=1 arm (same method), per expert + geomean over experts."""
import json, glob, sys, math
import numpy as np
HERE = "/home/coder/git/nestquant/threads/12-reference-encoder"
COLS = [("val", "all/forced"), ("val", "all/routed"), ("val", "bnd:think/d<=32/forced"), ("val", "bnd:think/d=1/forced"),
        ("val", "bnd:end/d<=32/forced"), ("val", "bnd:end/d=1/forced"), ("val", "bnd:end/d<=32/routed"),
        ("matched", "all/routed"), ("matched", "ood/forced"), ("matched", "ood/routed")]
SHORT = ["valF", "valR", "thk32F", "thk1F", "end32F", "end1F", "end32R", "mR", "mOodF", "mOodR"]
METHODS = ["nq/L4", "EXL3-4", "nq/L2", "EXL3-2"]
files = sorted(glob.glob(f"{HERE}/results_bnd/L*_E*.json"))
R = {f.split("/")[-1][:-5]: json.load(open(f)) for f in files}
arms = []
for r in R.values():
    for k in r["info"]:
        if k not in arms:
            arms.append(k)
agg = {}
for m in METHODS:
    print(f"\n=== {m}: % vs bnd1 (same method)   cols: {' '.join(SHORT)}")
    for name, r in R.items():
        base = {c: r["eval"][c[0]].get(f"{m}@bnd1", {}).get(c[1]) for c in COLS}
        for arm in arms:
            if arm == "bnd1" or f"{m}@{arm}" not in r["eval"]["val"]:
                continue
            vals = []
            for c in COLS:
                v = r["eval"][c[0]].get(f"{m}@{arm}", {}).get(c[1])
                vals.append(100 * (v / base[c] - 1) if v and base[c] else float("nan"))
                agg.setdefault((m, arm, c), []).append(v / base[c] if v and base[c] else float("nan"))
            print(f"  {name:<9} {arm:<12} " + " ".join(f"{x:+6.2f}" for x in vals))
    for arm in arms:
        if arm == "bnd1" or (m, arm, COLS[0]) not in agg:
            continue
        g = [100 * (math.exp(np.nanmean(np.log(agg[(m, arm, c)]))) - 1) for c in COLS]
        print(f"  {'GEOMEAN':<9} {arm:<12} " + " ".join(f"{x:+6.2f}" for x in g) + f"  (n={len(agg[(m, arm, COLS[0])])})")
print("\n=== nq/L4 vs EXL3-4 (same arm, same H), %:  cols " + " ".join(SHORT))
for name, r in R.items():
    for arm in arms:
        if f"nq/L4@{arm}" not in r["eval"]["val"]:
            continue
        v = [100 * (r["eval"][c[0]][f"nq/L4@{arm}"][c[1]] / r["eval"][c[0]][f"EXL3-4@{arm}"][c[1]] - 1)
             if c[1] in r["eval"][c[0]].get(f"nq/L4@{arm}", {}) else float("nan") for c in COLS]
        print(f"  {name:<9} {arm:<12} " + " ".join(f"{x:+6.2f}" for x in v))
print("\n=== damping (effective routed weights per group) ")
for name, r in R.items():
    for arm, inf in r["info"].items():
        d = inf["meta"].get("bnd_damp")
        if d:
            g = d["groups"]
            print(f"  {name:<9} {arm:<12} scale {d['scale']:.3f} share(A0) {d['trace_share_undamped_by_cap'].get('A0', 0):.3f} "
                  + " ".join(f"{k.split(':')[0][0]}{k.split(':')[1]}:{v['w_eff_routed']:.1f}(ess{v['ess']:.0f})" for k, v in g.items()))
