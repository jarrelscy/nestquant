"""H-structure correlates for dist48: per expert/projection max diag/mean diag, top-1 eigen share of H (input Gram)."""
import os, sys, json, glob
import torch
sys.path.insert(0, "/home/coder/git/nestquant/threads/19-full-capture")
import nq19_load
torch.cuda.set_per_process_memory_fraction(12 / 80)
cap = nq19_load.Capture()
out = {}
for f in sorted(glob.glob("results_dist48/L*_E*.json")):
    r = json.load(open(f)); L, E = r["layer"], r["expert"]
    HG = cap.glm_H(L, E)
    s = {}
    for pn, H in (("x", HG["H"][0]), ("a", HG["H"][2])):
        H = H.cuda().double(); d = H.diagonal()
        ev = torch.linalg.eigvalsh(H).clamp_min(0)
        top = torch.topk(d, 4)
        s[pn] = dict(maxdiag=float(top.values[0] / d.mean()), top_ch=top.indices.tolist(),
                     top1=float(ev[-1] / ev.sum()), effrank=float(ev.sum() ** 2 / (ev ** 2).sum()))
    ev = r["eval"]
    g = lambda m: ev[m]["all/routed"]
    s["L2gap"] = 100 * (g("nq/L2") / g("EXL3-2") - 1); s["L4gap"] = 100 * (g("nq/L4") / g("EXL3-4") - 1)
    s["L2gapF"] = 100 * (ev["nq/L2"]["all/forced"] / ev["EXL3-2"]["all/forced"] - 1)
    out[f"{L}:{E}"] = s
    print(f"{L}:{E:<4d} L2 {s['L2gap']:+7.1f} L4 {s['L4gap']:+6.1f} L2F {s['L2gapF']:+6.1f} | a maxdiag {s['a']['maxdiag']:7.1f} top1 {s['a']['top1']:.3f} | x maxdiag {s['x']['maxdiag']:6.1f} top1 {s['x']['top1']:.3f} ch {s['x']['top_ch'][:2]}", flush=True)
    del HG; torch.cuda.empty_cache()
json.dump(out, open("results_dist48/struct.json", "w"), indent=1)
