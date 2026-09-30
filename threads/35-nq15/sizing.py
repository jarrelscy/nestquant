"""T35 sizing: real NestQuant per-expert bits (all 75 layer manifests) -> hot count H that fits a routed-expert budget
at base rate b (pattern-rate base; overhead of the base = the real L2 overhead above 2.0, carried over unchanged)."""
import json, glob, sys
import numpy as np
R = "/tmp/nestquant/src/nq-hf/layers"
P = {"gate": 2048 * 6144, "up": 2048 * 6144, "down": 6144 * 2048}
b2, b4, n = 0.0, 0.0, 0
for f in sorted(glob.glob(f"{R}/L*/manifest.json")):
    m = json.load(open(f))
    for e, pe in m["per_expert"].items():
        for p, np_ in P.items():
            b = pe["proj"][p]["bits"]
            b2 += b["2"] * np_; b4 += b["4"] * np_
        n += 1
tot = sum(P.values()) * n
b2, b4 = b2 / tot, b4 / tot
GiB = 2 ** 30
per_bit_all = sum(P.values()) * n / 8 / GiB           # GiB per bpw over all routed experts
per_bit_slot = sum(P.values()) * n / 256 / 8 / GiB    # GiB per bpw for one slot in every layer
print(f"experts {n} (= {n // 256} layers x 256)  L2 bpw {b2:.4f}  L4 bpw {b4:.4f}  (residual {b4 - b2:.4f})")
print(f"GiB per bpw: all experts {per_bit_all:.2f}; one slot x all layers {per_bit_slot:.4f}")
BUD = float(sys.argv[1]) if len(sys.argv) > 1 else 175.0
ov = b2 - 2.0
out = {"b2": b2, "b4": b4, "per_bit_all": per_bit_all, "per_bit_slot": per_bit_slot, "budget": BUD, "rows": []}
for b in (1.0, 1.25, 1.5, 1.75, 2.0):
    base = b + ov
    for tot4, lab in ((b4, "L4 total fixed at real 4.14"), (4.0 + ov, "4.0+ov (prompt)")):
        extra = (tot4 - base) * per_bit_slot
        H = (BUD - base * per_bit_all) / extra
        out["rows"].append(dict(b=b, base_bpw=base, top=tot4, H=H, base_GiB=base * per_bit_all, extra_per_slot=extra))
        print(f"b={b:.2f} base {base:.3f} bpw = {base * per_bit_all:6.1f} GiB | hot total {tot4:.3f}: +{extra:.3f} GiB/slot"
              f" -> H = {H:6.1f}   [{lab}]")
json.dump(out, open("/tmp/nestquant/35-nq15/sizing.json", "w"), indent=1)
