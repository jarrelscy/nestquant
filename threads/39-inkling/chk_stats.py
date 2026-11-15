import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nq_ink_env as E
import nq_layer as NL
cap = NL.open_stats(sys.argv[1])
L = int(sys.argv[2])
import json; m = json.load(open(f"{sys.argv[1]}/stats/L{L}/meta.json"))
nr = m["n_routed"]; top = sorted(range(384), key=lambda e: -nr[e])[:3]
print("T_fit", m["T_fit"], "routed rows top", [(e, nr[e]) for e in top], "zero-routed", sum(1 for v in nr if v == 0))
for e in top[:2]:
    HG, fl = NL.expert_HG(cap, L, e)
    for i, H in enumerate(HG["H"]):
        ev = torch.linalg.eigvalsh(H.double())
        print(e, i, tuple(H.shape), "sym", float((H - H.T).abs().max()), "eig min/max", float(ev.min()), float(ev.max()), "tr", float(H.trace()))
    print("G", [None if g is None else tuple(g.shape) for g in HG["G"]], fl)
