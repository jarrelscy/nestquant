"""lambda-fallback A/B summary (results_lam): arm a = lam 0.3 everywhere, arm b = lam 0 where the v1 routed L2 of (a)
exceeds EXL3-2 by > 1.5 % (decided per seed), deltas vs EXL3 of the same seed; val = T19 pooled held-out rows."""
import json, glob
import numpy as np
HERE = "/home/coder/git/nestquant/threads/12-reference-encoder"
COLS = [("v1", "all/routed"), ("v1", "ood/forced"), ("v1", "ood/routed"), ("val", "all/forced"), ("val", "all/routed")]
SH = ["routed", "oodF", "oodR", "valF", "valR"]
R = [json.load(open(f)) for f in sorted(glob.glob(f"{HERE}/results_lam/L*_E*.json"))]
seeds = sorted({k.split("@s")[1] for r in R for k in r["val"]})
out = {}
for r in R:
    nm = f"{r['layer']}:{r['expert']}"
    for s in seeds:
        g = lambda sp, m, c: r[sp].get(f"{m}@s{s}", {}).get(c)
        if g("v1", "nq_lam0/L2", "all/routed") is None:
            continue
        rel = lambda m, Lv, c: 100 * (g(c[0], f"{m}/L{Lv}", c[1]) / g(c[0], f"EXL3-{Lv}", c[1]) - 1)
        fb = rel("nq_lam0.3", 2, COLS[0]) > 1.5
        for arm, m in (("a", "nq_lam0.3"), ("lam0", "nq_lam0"), ("b", "nq_lam0" if fb else "nq_lam0.3")):
            for Lv in (2, 4):
                out[(nm, s, arm, Lv)] = [rel(m, Lv, c) for c in COLS]
        out[(nm, s, "fb")] = fb
names = sorted({k[0] for k in out})
print("per expert, seed: L2 a | L2 lam0 | L4 a | L4 lam0   (cols " + "/".join(SH) + ")")
for nm in names:
    for s in seeds:
        if (nm, s, "fb") not in out:
            continue
        f = lambda arm, Lv: " ".join(f"{x:+5.2f}" for x in out[(nm, s, arm, Lv)])
        print(f"  {nm:<7} s{s:<6} {'FB' if out[(nm, s, 'fb')] else '  '} L2a {f('a', 2)} | L2λ0 {f('lam0', 2)} | L4a {f('a', 4)} | L4λ0 {f('lam0', 4)}")
for s in seeds:
    ks = [nm for nm in names if (nm, s, "fb") in out]
    fbs = [nm for nm in ks if out[(nm, s, "fb")]]
    print(f"\nseed {s}: n={len(ks)}, fallback experts {fbs}")
    for grp, lst in (("all", ks), ("fallback", fbs)):
        if not lst:
            continue
        for arm in ("a", "b"):
            for Lv in (2, 4):
                x = np.array([out[(nm, s, arm, Lv)] for nm in lst])
                print(f"  {grp:<9} arm {arm} L{Lv} mean " + " ".join(f"{v:+5.2f}" for v in x.mean(0)) +
                      "   worst " + " ".join(f"{v:+5.2f}" for v in x.max(0)))
