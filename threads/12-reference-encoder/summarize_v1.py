"""Tables for the v1 sweep: per-expert metrics, deltas vs EXL3-4 / matched EXL3, 9-expert mean and worst."""
import json, glob, os, sys
D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_v1")
KEYS = ("all/routed", "ood/forced", "ood/routed")
books = {os.path.basename(f)[:-5]: json.load(open(f)) for f in sorted(glob.glob(f"{D}/L*_E*.json"))}
names = []
for R in books.values():
    for n in R["eval"]:
        if n not in names:
            names.append(n)
print("experts:", list(books))
for ex, R in books.items():
    print(f"\n== {ex}")
    ref = R["eval"].get("EXL3-4")
    for n in names:
        if n not in R["eval"]:
            continue
        v = R["eval"][n]; inf = R.get("info", {}).get(n, {})
        d = " ".join(f"{100 * (v[k] / ref[k] - 1):+6.1f}%" for k in KEYS) if ref and "L2" not in n and "-2" not in n else ""
        print(f"  {n:22s} {inf.get('bpw', 0):.4f}  " + " ".join(f"{v[k]:7.3f}" for k in KEYS) + f"   {d}  "
              + (f"L2=base:{inf['L2_equals_base']}" if "L2_equals_base" in inf else ""))


def agg(name, anchor):
    out = []
    for k in KEYS:
        rs = [R["eval"][name][k] / R["eval"][anchor][k] - 1 for R in books.values() if name in R["eval"] and anchor in R["eval"]]
        if rs:
            out.append((k, 100 * sum(rs) / len(rs), 100 * max(rs), len(rs)))
    return out


print("\n== 9-expert summary: mean / worst delta (%)")
for anchor in ("EXL3-4", "EXL3-2"):
    print(f" vs {anchor}")
    for n in names:
        is2 = n.endswith("/L2") or n == "EXL3-2"
        if (anchor == "EXL3-2") != is2 or n == anchor:
            continue
        a = agg(n, anchor)
        if a:
            print(f"  {n:22s} " + "  ".join(f"{k}: {m:+5.2f} / {w:+5.2f}" for k, m, w, c in a) + f"  (n={a[0][3]})")
print(" vs matched EXL3 (same bpw)")
for T in ("4.0625", "4.09375", "4.125", "4.25"):
    a = agg(f"nq_r{T}/L4", f"EXL3-4+{T}")
    if a:
        print(f"  nq_r{T:8s} vs EXL3-4+{T:8s}" + "  ".join(f"{k}: {m:+5.2f} / {w:+5.2f}" for k, m, w, c in a) + f" (n={a[0][3]})")
a = agg("nq/L4", "EXL3-4+4.0221")
if a:
    print(f"  nq/L4 (4.0221) vs EXL3-4+4.0221 " + "  ".join(f"{k}: {m:+5.2f} / {w:+5.2f}" for k, m, w, c in a))
