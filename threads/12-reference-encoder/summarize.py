"""Tabulate results/L*_E*.json: routed/forced/OOD % + bpw per method; closer deltas; smallest extra bpw beating EXL3-4."""
import json, glob, os, sys
RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
books = {}
for f in sorted(glob.glob(f"{RES}/L*_E*.json")):
    if "withL3" in f: continue
    books[os.path.basename(f)[:-5]] = json.load(open(f))
cols = ["all/routed", "all/forced", "ood/routed", "ood/forced"]
for name, R in books.items():
    print(f"\n== {name}")
    ev, inf = R.get("eval", {}), R.get("info", {})
    for m, v in ev.items():
        if "/L3" in m or m == "EXL3-3": continue
        bpw = inf.get(m, {}).get("bpw")
        print(f"  {m:28s} " + " ".join(f"{v.get(c, float('nan')):7.3f}" for c in cols) + (f"  bpw {bpw:.4f}" if bpw else ""))
# gap table at level 4
print("\n== level-4 vs EXL3-4 (routed, ood/routed deltas %, relative)")
meths = sorted({m for R in books.values() for m in R.get("eval", {}) if m.endswith("/L4")})
agg = {}
for m in meths:
    row = []
    for name, R in books.items():
        ev = R["eval"]
        if m not in ev or "EXL3-4" not in ev: row.append(None); continue
        a, b = ev[m], ev["EXL3-4"]
        row.append((a["all/routed"] / b["all/routed"] - 1, a["ood/routed"] / b["ood/routed"] - 1, R["info"][m]["bpw"]))
    agg[m] = row
    ok = [r for r in row if r]
    beats = all(r[0] < 0 and r[1] < 0 for r in ok)
    print(f"  {m:28s} n={len(ok)} bpw {sum(r[2] for r in ok)/max(len(ok),1):.4f}  routed {100*sum(r[0] for r in ok)/max(len(ok),1):+6.2f}%  "
          f"ood {100*sum(r[1] for r in ok)/max(len(ok),1):+6.2f}%  worst r {100*max((r[0] for r in ok), default=0):+6.2f}% o {100*max((r[1] for r in ok), default=0):+6.2f}%  beats-all {beats}")
print("\n== level-2 vs EXL3-2")
for m in sorted({m for R in books.values() for m in R.get("eval", {}) if m.endswith("/L2")}):
    ok = [(R["eval"][m]["all/routed"] / R["eval"]["EXL3-2"]["all/routed"] - 1, R["eval"][m]["ood/routed"] / R["eval"]["EXL3-2"]["ood/routed"] - 1)
          for R in books.values() if m in R.get("eval", {}) and "EXL3-2" in R["eval"]]
    print(f"  {m:28s} n={len(ok)} routed {100*sum(r[0] for r in ok)/len(ok):+6.2f}%  ood {100*sum(r[1] for r in ok)/len(ok):+6.2f}%  worst r {100*max(r[0] for r in ok):+6.2f}%")
