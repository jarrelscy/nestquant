"""T26 post-run check: blend finite / trace-1 on a zero-vision expert and normal experts; fixed-set score sums to 1."""
import sys, torch, numpy as np
import nq19_load as C, nq26_blend as B
t = C.Capture(root="/tmp/nestquant/19-capture-glmfmt")
v = C.Capture(root="/tmp/nestquant/19-capture-mm")
cap = B.BlendCapture(t, v, 0.25)
for L, E in [(15, 50), (15, 0), (40, 7), (77, 100)]:
    r = cap.glm_H(L, E, device="cpu")
    tr = [float(torch.diagonal(h).mean()) for h in r["H"]]
    fin = all(bool(torch.isfinite(h).all()) for h in r["H"]) and all(g is None or bool(torch.isfinite(g).all()) for g in r["G"])
    print(L, E, "w_eff", r["meta"]["w_vision_eff"], "n_v", r["meta"]["vision"]["n_routed"], "finite", fin, "meandiag", [round(x, 4) for x in tr])
for L in (3, 40, 77):
    s = cap.fixed_set_score(L)
    print("fixed_set_score", L, float(s.sum()), bool(np.isfinite(s).all()))
