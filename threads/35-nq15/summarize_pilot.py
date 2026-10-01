"""Summarise the 9-expert Stage 1 pilot (results/pilot/L*_E*.json) -> results/pilot_summary.json + printed tables.

rel-L2 = harness router-weighted relative L2 (%), keys {all,control,ood}/{routed,forced}.
Comparisons: b20/L2 vs EXL3-2, b175/L2 vs EXL3-1.75, b15/L2 vs EXL3-1.5; every L4 vs EXL3-4 and EXL3-4+4.125.
"""
import os, json, glob, math
HERE = os.path.dirname(os.path.abspath(__file__))
KEYS = ["all/routed", "all/forced", "control/routed", "ood/routed"]
ROWS = ["EXL3-1.5", "EXL3-1.75", "EXL3-2", "nq15_b15/L2", "nq15_b175/L2", "nq15_b20/L2",
        "EXL3-4", "EXL3-4+4.125", "nq15_b15/L4", "nq15_b175/L4", "nq15_b20/L4"]
PAIRS = [("nq15_b20/L2", "EXL3-2"), ("nq15_b175/L2", "EXL3-1.75"), ("nq15_b15/L2", "EXL3-1.5"),
         ("nq15_b175/L2", "nq15_b20/L2"), ("nq15_b15/L2", "nq15_b20/L2"),
         ("nq15_b20/L4", "EXL3-4"), ("nq15_b175/L4", "EXL3-4"), ("nq15_b15/L4", "EXL3-4"),
         ("nq15_b20/L4", "EXL3-4+4.125"), ("nq15_b175/L4", "EXL3-4+4.125"), ("nq15_b15/L4", "EXL3-4+4.125"),
         ("nq15_b175/L4", "nq15_b20/L4"), ("nq15_b15/L4", "nq15_b20/L4")]


def main():
    files = sorted(glob.glob(f"{HERE}/results/pilot/L*_E*.json"))
    D = {os.path.basename(f)[:-5]: json.load(open(f)) for f in files}
    out = dict(n_experts=len(D), experts=list(D), rows={}, pairs={}, direction={}, checks={})
    for r in ROWS:
        v = {k: [d["eval"][r][k] for d in D.values() if r in d["eval"]] for k in KEYS}
        if not v[KEYS[0]]:
            continue
        bpw = [d["info"][r].get("bpw") for d in D.values() if r in d.get("info", {}) and d["info"][r].get("bpw")]
        out["rows"][r] = dict(n=len(v[KEYS[0]]), mean={k: sum(x) / len(x) for k, x in v.items()},
                              bpw=sum(bpw) / len(bpw) if bpw else None)
    for a, b in PAIRS:
        d = [100 * (x["eval"][a]["all/routed"] / x["eval"][b]["all/routed"] - 1)
             for x in D.values() if a in x["eval"] and b in x["eval"]]
        if d:
            out["pairs"][f"{a} vs {b}"] = dict(n=len(d), mean_pct=sum(d) / len(d), worst_pct=max(d), best_pct=min(d))
    for cfg in ("b175", "b15"):
        for Lv in ("L2", "L4"):
            rs = [x["direction"][cfg][Lv]["all"] for x in D.values() if cfg in x.get("direction", {})]
            if rs:
                out["direction"][f"{cfg}/{Lv}"] = {k: dict(mean=sum(r[k] for r in rs) / len(rs), min=min(r[k] for r in rs),
                                                            max=max(r[k] for r in rs)) for k in rs[0]}
    ok = True
    for n, x in D.items():
        for r, inf in x["info"].items():
            if r.startswith("nq15_"):
                ok &= all(inf["roundtrip_bitexact"].values()) and all(v == 0 for v in inf["ref15_mismatch"].values())
                ok &= all(inf["bitexact_internal"].values())
    out["checks"]["all_roundtrip_ref15_bitexact"] = bool(ok)
    out["checks"]["encode_time_s"] = {c: sum(x["info"][f"nq15_{c}/L2"]["time_s"] for x in D.values()
                                             if f"nq15_{c}/L2" in x["info"]) / len(D) for c in ("b20", "b175", "b15")}
    json.dump(out, open(f"{HERE}/results/pilot_summary.json", "w"), indent=1)
    print(f"{len(D)} experts: {list(D)}")
    print(f"{'row':16s} {'bpw':>7s} " + " ".join(f"{k:>15s}" for k in KEYS))
    for r, v in out["rows"].items():
        print(f"{r:16s} {v['bpw'] or float('nan'):7.4f} " + " ".join(f"{v['mean'][k]:15.3f}" for k in KEYS))
    print("\npairs (all/routed rel-L2, % change of first vs second):")
    for k, v in out["pairs"].items():
        print(f"  {k:36s} mean {v['mean_pct']:+6.2f}%  worst {v['worst_pct']:+6.2f}%  best {v['best_pct']:+6.2f}%  n={v['n']}")
    print("\ndirection (E_b vs E_b20, overall):")
    for k, v in out["direction"].items():
        print("  " + k + "  " + "  ".join(f"{m} {s['mean']:.3f} [{s['min']:.3f},{s['max']:.3f}]" for m, s in v.items()))
    print("\nchecks", out["checks"])


if __name__ == "__main__":
    main()
