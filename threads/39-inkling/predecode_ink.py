"""Decode one encoded layer's experts at levels 2 and 4 -> OUT/L{L}.lv{2,4}.pt {gu bf16 [E,2F,D], dn bf16 [E,D,F]}
(gu rows = [gate; up], matching inkdq.LayerExperts.w).  Also writes OUT/L{L}.relerr.json: weight-space rel. error per
expert/level vs the native source."""
import os, sys, json, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nq_ink_env as E
ND = E.ND
root, out, L = sys.argv[1], sys.argv[2], int(sys.argv[3])
os.makedirs(out, exist_ok=True)
if all(os.path.exists(f"{out}/L{L}.lv{v}.pt") for v in (2, 4)):
    sys.exit(0)
gu = {v: torch.empty(E.NEXP, 2 * E.F, E.D, dtype=torch.bfloat16) for v in (2, 4)}
dn = {v: torch.empty(E.NEXP, E.D, E.F, dtype=torch.bfloat16) for v in (2, 4)}
src = E.Q.LayerExperts(L, "cuda")
rel = {2: [], 4: []}
for e in range(E.NEXP):
    art = torch.load(f"{root}/L{L}/experts/E{e}.pt", weights_only=False, map_location="cuda")
    sg, sd = src.w(e, torch.float32)
    for v in (2, 4):
        g, u, d = ND.decode_expert(art, v)
        w = torch.cat([g, u], 0)
        gu[v][e] = w.bfloat16().cpu(); dn[v][e] = d.bfloat16().cpu()
        rel[v].append([float((w - sg).norm() / sg.norm()), float((d - sd).norm() / sd.norm())])
for v in (2, 4):
    torch.save(dict(gu=gu[v], dn=dn[v]), f"{out}/L{L}.lv{v}.pt.tmp"); os.replace(f"{out}/L{L}.lv{v}.pt.tmp", f"{out}/L{L}.lv{v}.pt")
json.dump(rel, open(f"{out}/L{L}.relerr.json", "w"))
print(L, {v: [round(sum(r[i] for r in rel[v]) / len(rel[v]), 4) for i in (0, 1)] for v in (2, 4)}, flush=True)
