"""Eigen-plane direction stability: share of each calibration direction v (NE.lr_detect on glm_H) in held-out Grams
(T19 val and matched captures; forced = all rows, routed = routed rows weighted by p) vs its calibration share."""
import sys, json, glob
import torch
sys.path.insert(0, "/home/coder/git/nestquant/threads/19-full-capture")
import nq19_load, nq_encode as NE
torch.cuda.set_per_process_memory_fraction(12 / 80)
torch.backends.cuda.matmul.allow_tf32 = False
cap = nq19_load.Capture()
out = {}
for f in sorted(glob.glob("results_dist48/L*_E*.json")):
    r = json.load(open(f)); L, E = r["layer"], r["expert"]
    HG = cap.glm_H(L, E)
    Vs = {"x": NE.lr_detect(HG["H"][0].cuda()).float(), "a": NE.lr_detect(HG["H"][2].cuda()).float()}
    cal = {k: [float(v @ HG["H"][0 if k == "x" else 2].cuda().float() @ v / HG["H"][0 if k == "x" else 2].diagonal().sum()) for v in V]
           for k, V in Vs.items()}
    res = dict(cal=cal)
    for split in ("val", "matched"):
        dm = cap.expert_data(L, E, split); c = dm.capture
        x = c["x"].cuda().float()
        Wg, Wu = dm.teacher[0], dm.teacher[1]
        routed, slots = torch.where(c["ids"] == E)
        p = c["p"][routed, slots].cuda().float()
        for kind, rows, w in (("forced", torch.arange(len(x), device="cuda"), None), ("routed", routed.cuda(), p)):
            xs = x[rows]
            a = torch.nn.functional.silu(xs @ Wg.T) * (xs @ Wu.T)
            for k, Z in (("x", xs), ("a", a)):
                ww = torch.ones(len(Z), device="cuda") if w is None else w
                tr = float((ww[:, None] * Z.square()).sum())
                res[f"{split}/{kind}/{k}"] = [float((ww * (Z @ v).square()).sum()) / tr for v in Vs[k]]
        del dm, c, x
    out[f"{L}:{E}"] = res
    fmt = lambda l: "[" + ",".join(f"{v:.3f}" for v in l) + "]"
    print(f"{L}:{E:<4d} x cal {fmt(cal['x'])} val F {fmt(res['val/forced/x'])} R {fmt(res['val/routed/x'])} | "
          f"a cal {fmt(cal['a'])} val F {fmt(res['val/forced/a'])} R {fmt(res['val/routed/a'])} matched R {fmt(res['matched/routed/a'])}", flush=True)
    torch.cuda.empty_cache()
json.dump(out, open("results_dist48/lr_stability.json", "w"), indent=1)
