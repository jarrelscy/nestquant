"""Stage-2 reproduction check vs orbit pilot statistics (grams) and thread-12 glm_H cache (H, G)."""
import sys, json, torch
import nq19_load as C
root = sys.argv[1]; L = int(sys.argv[2]); experts = [int(v) for v in sys.argv[3].split(",")]
cap = C.Capture(root)
ORB = "/home/coder/git/orbit-duet/runs"
rel = lambda a, b: float((a.double() - b.double()).norm() / b.double().norm())
mx = lambda a, b: float(((a.double() - b.double()).abs().max() / b.double().abs().max()))
res = {}
for E in experts:
    st = torch.load(f"{ORB}/glm53_pilot_matched_l{L}/statistics/l{L}_e{E}.pt", weights_only=True, mmap=True)
    mine = cap.pilot_stats(L, E)
    r = dict(rows=[mine["metadata"]["training_rows"], st["metadata"]["training_rows"]],
             mass=[mine["metadata"]["mass"], st["metadata"]["mass"]],
             W_x_rel=rel(mine["grams"][0], st["grams"][0].cuda()), W_x_maxrel=mx(mine["grams"][0], st["grams"][0].cuda()),
             W_down_rel=rel(mine["grams"][1], st["grams"][1].cuda()), W_down_maxrel=mx(mine["grams"][1], st["grams"][1].cuda()))
    HG = cap.glm_H(L, E)
    try:
        ref = torch.load(f"/tmp/nestquant/12-reference-encoder/H_l{L}_e{E}.pt")
        for i, n in enumerate(("gate", "up", "down")):
            r[f"H_{n}_rel"] = rel(HG["H"][i], ref["H"][i].cuda())
        for i, n in enumerate(("gate", "up")):
            r[f"G_{n}_rel"] = rel(HG["G"][i].diagonal(), ref["G"][i].cuda().diagonal())
    except FileNotFoundError:
        r["t12_cache"] = "missing"
    r["ess"] = HG["meta"]["ess"]
    res[E] = r
print(json.dumps(res, indent=1))
