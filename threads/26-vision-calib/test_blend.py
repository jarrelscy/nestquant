import numpy as np, torch, json
import nq19_load as C, nq26_blend as B
t = C.Capture(root="/tmp/nestquant/19-capture-glmfmt", stats="stats0")
v = C.Capture(root="/tmp/nestquant/19-capture-mm-test")
cap = B.BlendCapture(t, v, 0.25)
for L in (3, 4):
    sv = v.salience(L)["sums"][:, 0]; st = t.salience(L)["sums"][:, 0]
    n_v, n_t = sv[:, 0], st[:, 0]
    rho = np.corrcoef(n_v / n_v.sum(), n_t / n_t.sum())[0, 1]
    from scipy.stats import spearmanr
    rs = spearmanr(sv[:, 4], st[:, 4]).correlation
    print(f"L{L}: vision routed rows total {int(n_v.sum())} (= 8 x {int(n_v.sum()//8)}), experts with 0 vision rows {int((n_v==0).sum())}, "
          f"usage corr {rho:.3f}, REAP-sum spearman {rs:.3f}")
    for E in (0, int(np.argmax(n_v)), int(np.argmin(n_v))):
        hb = cap.glm_H(L, E); ht = t.glm_H(L, E); hv = v.glm_H(L, E)
        rel = lambda a, b: float((a - b).norm() / b.norm())
        print(f"  E{E}: n_vis {int(n_v[E])} n_text {int(n_t[E])} | |Hv-Ht|/|Ht| x {rel(hv['H'][0], ht['H'][0]):.3f} down {rel(hv['H'][2], ht['H'][2]):.3f}"
              f" | |Hb-Ht|/|Ht| x {rel(hb['H'][0], ht['H'][0]):.3f} down {rel(hb['H'][2], ht['H'][2]):.3f} | trace/n x {float(torch.diagonal(hb['H'][0]).mean()):.4f}"
              f" finite {all(torch.isfinite(h).all().item() for h in hb['H'])} G {[None if g is None else round(float(torch.diagonal(g).pow(2).mean()),4) for g in hb['G']]}")
    s = cap.fixed_set_score(L); print("  fixed-set blend score sum", round(float(s.sum()), 6), "top5", np.argsort(-s)[:5].tolist())
