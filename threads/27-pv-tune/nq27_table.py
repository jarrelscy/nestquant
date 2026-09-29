"""T27 per-expert table on eval/val: L2/L4 before -> after (all/routed + all/forced), vs EXL3, per-layer means.

  python nq27_table.py ARM [--key all/routed] [--md]
Relative deltas in %: tuned vs untuned (d = after/before - 1) and vs EXL3-K (nq/EXL3 - 1, before and after).
"""
import json, glob, sys, argparse, os
import numpy as np

OUT = "/tmp/nestquant/27-pv-tune"


def rel(a, b):
    return 100 * (a / b - 1)


def pick(spec, L, E):
    """'sel:A,B' -> per expert the arm with the best chunk-0 HELD-OUT level-balanced loss (no eval/val peeking)."""
    if not spec.startswith("sel:"):
        return spec
    best = None
    for arm in spec[4:].split(","):
        rc = json.load(open(f"{OUT}/arms/{arm}/L{L}/experts/E{E}.json"))
        if not os.path.exists(f"{OUT}/val/{arm}/L{L}_E{E}.json"):
            raise FileNotFoundError(arm)
        j = sum(0.5 * (rc["held_best"][l] / rc["held0"][l]) ** 2 for l in ("2", "4"))
        if best is None or j < best[0]:
            best = (j, arm)
    return best[1]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("arm"); ap.add_argument("--key", default="all/routed")
    ap.add_argument("--layers", default="3,4,5,6,30")
    a = ap.parse_args()
    ex = json.load(open("/home/coder/git/nestquant/threads/27-pv-tune/experts.json"))
    k = a.key
    print(f"arm {a.arm}, key {k} (eval/val, relative router-weighted output L2 %)\n")
    hdr = ("| L | E | set | val rows | ESS val | ESS held | nq L2 before | after | d L2 | EXL3-2 | vs EXL3 before | after "
           "| nq L4 before | after | d L4 | EXL3-4 | vs EXL3 before | after | best step | tuned |")
    print(hdr); print("|" + "---|" * (hdr.count("|") - 1))
    allrows = []
    for L in map(int, a.layers.split(",")):
        e = ex[str(L)]; rows = []
        for E in e["flagged"] + e["random"]:
            try:
                b = json.load(open(f"{OUT}/val/base/L{L}_E{E}.json"))["eval"]
                arm = pick(a.arm, L, E)
                t = json.load(open(f"{OUT}/val/{arm}/L{L}_E{E}.json"))["eval"]
                rc = json.load(open(f"{OUT}/arms/{arm}/L{L}/experts/E{E}.json"))
                rc["tune"] = rc["tune"] if arm == a.arm else f"{arm}"
            except FileNotFoundError:
                continue
            try:
                es = json.load(open(f"{OUT}/val/ess/L{L}_E{E}.json")); ev_, eh_ = es["val"]["ess"], es["held"]["ess"]
            except FileNotFoundError:
                ev_ = eh_ = float("nan")
            r = dict(L=L, E=E, set="flag" if E in e["flagged"] else "rand", n=e["val_rows"][str(E)], ev=ev_, eh=eh_,
                     b2=b["nq/L2"][k], t2=t["L2"][k], x2=b["EXL3-2"][k], b4=b["nq/L4"][k], t4=t["L4"][k], x4=b["EXL3-4"][k],
                     step=rc["best_step"], tune=rc["tune"])
            rows.append(r)
            print(f"| {L} | {E} | {r['set']} | {r['n']} | {r['ev']:.0f} | {r['eh']:.0f} | {r['b2']:.3f} | {r['t2']:.3f} | {rel(r['t2'], r['b2']):+.2f} | {r['x2']:.3f} "
                  f"| {rel(r['b2'], r['x2']):+.2f} | {rel(r['t2'], r['x2']):+.2f} | {r['b4']:.3f} | {r['t4']:.3f} | {rel(r['t4'], r['b4']):+.2f} "
                  f"| {r['x4']:.3f} | {rel(r['b4'], r['x4']):+.2f} | {rel(r['t4'], r['x4']):+.2f} | {r['step']} | {r['tune']} |")
        if rows:
            m = lambda f: np.mean([f(r) for r in rows])
            print(f"| **L{L} mean** | {len(rows)} | | | | | | | **{m(lambda r: rel(r['t2'], r['b2'])):+.2f}** | | {m(lambda r: rel(r['b2'], r['x2'])):+.2f} "
                  f"| {m(lambda r: rel(r['t2'], r['x2'])):+.2f} | | | **{m(lambda r: rel(r['t4'], r['b4'])):+.2f}** | | {m(lambda r: rel(r['b4'], r['x4'])):+.2f} "
                  f"| {m(lambda r: rel(r['t4'], r['x4'])):+.2f} | | |")
        allrows += rows
    early = [r for r in allrows if r["L"] != 30]
    for name, rs in (("L3-L6", early), ("L3-L6 flagged", [r for r in early if r["set"] == "flag"]),
                     ("L3-L6 random", [r for r in early if r["set"] == "rand"]),
                     ("L3-L6 ESS_val>=100", [r for r in early if r["ev"] >= 100]),
                     ("L3-L6 ESS_val<100", [r for r in early if r["ev"] < 100]),
                     ("L30 control", [r for r in allrows if r["L"] == 30])):
        if rs:
            d2 = [rel(r["t2"], r["b2"]) for r in rs]; d4 = [rel(r["t4"], r["b4"]) for r in rs]
            print(f"\n{name}: n {len(rs)}  dL2 mean {np.mean(d2):+.2f}% (min {min(d2):+.2f}, max {max(d2):+.2f})  "
                  f"dL4 mean {np.mean(d4):+.2f}% (min {min(d4):+.2f}, max {max(d4):+.2f})  "
                  f"vs EXL3-2 {np.mean([rel(r['b2'], r['x2']) for r in rs]):+.2f} -> {np.mean([rel(r['t2'], r['x2']) for r in rs]):+.2f}  "
                  f"vs EXL3-4 {np.mean([rel(r['b4'], r['x4']) for r in rs]):+.2f} -> {np.mean([rel(r['t4'], r['x4']) for r in rs]):+.2f}")


if __name__ == "__main__":
    main()
