import sys, json
d = json.load(open(sys.argv[1]))
for n, v in d.items():
    t = v["table"]; m = v.get("meta") or {}
    b = m.get("bpw2" if n.endswith("@2") else "bpw4", "")
    print(f"{n:40s} {b!s:>6} " + " ".join(f"{t[k]:7.2f}" for k in ["all/forced","all/routed","control/forced","control/routed","ood/forced","ood/routed"]))
