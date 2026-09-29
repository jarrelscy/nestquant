"""T29 lead table: per expert val routed error % (L2 and L4) for shipped / rot / rot+lam0 / EXL3 / EXL3+rot, and the
% changes vs shipped and vs EXL3 (plain and +rot); means + worst for L3-6 and L7+.   python table29.py [--rot drot4]"""
import argparse
import numpy as np
from sum29 import load, EARLY, BAND, CTRL


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--rot", default="drot4"); a = ap.parse_args()
    R = load(); rot = a.rot
    for Lv in (2, 4):
        cols = ["art", rot, f"{rot}+lam0", f"X{Lv}", f"X{Lv}+{rot}"]
        print(f"\n### L{Lv}  (val routed rel. error %; d = % change)")
        print(f"| expert | shipped | {rot} | {rot}+lam0 | EXL3-{Lv} | EXL3-{Lv}+{rot} | {rot} d shipped | +lam0 d shipped | {rot} d EXL3 | +lam0 d EXL3 | {rot} d EXL3+rot |")
        print("|" + "---|" * 11)
        grp = {"L3-6": [], "L7+": []}
        for e in EARLY + BAND + CTRL:
            r = R.get(e, {})
            v = [r[c]["eval"][f"L{Lv}"]["routed"] if c in r else float("nan") for c in cols]
            d = [100 * (v[1] / v[0] - 1), 100 * (v[2] / v[0] - 1), 100 * (v[1] / v[3] - 1), 100 * (v[2] / v[3] - 1),
                 100 * (v[1] / v[4] - 1)]
            grp["L3-6" if int(e.split(":")[0]) <= 6 else "L7+"].append(d)
            print(f"| {e} | " + " | ".join(f"{x:.3f}" for x in v) + " | " + " | ".join(f"{x:+.1f}" for x in d) + " |")
        for g, ds in grp.items():
            D = np.array(ds); ok = ~np.isnan(D).any(1); D = D[ok]
            print(f"| **{g} mean (n={len(D)})** | | | | | | " + " | ".join(f"{x:+.1f}" for x in D.mean(0)) + " |")
            print(f"| **{g} worst** | | | | | | " + " | ".join(f"{x:+.1f}" for x in D.max(0)) + " |")


if __name__ == "__main__":
    main()
