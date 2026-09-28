"""T26: text/vision Hessian blend on top of T19's loader (no change to T19's files needed to use it).

    import nq19_load as C, nq26_blend as B
    cap = B.BlendCapture(C.Capture(root="/tmp/nestquant/19-capture-glmfmt"),          # text (production stats)
                         C.Capture(root="/tmp/nestquant/19-capture-mm"), w_vision=0.25)
    HG  = cap.glm_H(L, E)          # same dict as Capture.glm_H: {"H": [Hx, Hx, Ha], "G": [Gg, Gu, None], "meta"}
    S   = cap.fixed_set_score(L)   # blended per-expert REAP score for the fixed set

Blend (v3's 75/25 text/vision mix, applied at the point where thread 08's recipe produces H):
    H_p = (1 - w) H_text,p / mean(diag H_text,p) + w H_vis,p / mean(diag H_vis,p)      p in {gate/up, down}
    G_p = diag( (1 - w) g_text,p / mean(g_text,p) + w g_vis,p / mean(g_vis,p) )^1/2     (g = G^2, the dmix output)
Low-coverage experts: with n_v routed vision rows, the vision weight is w_e = w * min(1, n_v / n_min) (default
n_min = 128), and H/G are renormalised with (1 - w_e, w_e).  So an expert that no vision token routes to (its
p-weighted vision W is 0, so the vision H is undefined) gets exactly the text H, and one with a handful of rows is
not dominated by a noisy estimate.  meta["w_vision_eff"] records w_e.
Each side is T19's glm_H (thread-08 unif0.75, count 1) on its own capture, so both are already trace-normalised;
normalising again only makes the weights exact.  The vision U part includes T19's context rows (all mm tokens pushed
through every expert), but the p-weighted W part needs routed rows, hence the n_min ramp.  Encoder damping (sigma)
is added by the encoder afterwards, unchanged.  Boundary weighting (bnd_w) applies to the text side only.
All other Capture methods (eval_capture, expert_data, meta, ...) are delegated to the text capture.
"""
import numpy as np
import torch


def _nt(H):
    return H / torch.diagonal(H).mean()


class BlendCapture:
    def __init__(self, text, vision, w_vision=0.25, n_min=128):
        self.text, self.vision, self.w, self.n_min = text, vision, float(w_vision), n_min

    def __getattr__(self, k):
        return getattr(self.text, k)

    def glm_H(self, L, E, alpha=0.25, ctx_mass=0.25, device="cuda", bnd_w=None, **kw):
        ht = self.text.glm_H(L, E, alpha=alpha, ctx_mass=ctx_mass, device=device, bnd_w=bnd_w, **kw)
        if self.w == 0:
            return ht
        n_v = int(self.vision._arr(L, "scalars")[E][0])
        w = self.w * min(1.0, n_v / self.n_min) if self.n_min else (self.w if n_v else 0.0)
        if w == 0:
            return dict(ht, meta=dict(ht["meta"], w_vision=self.w, w_vision_eff=0.0, vision=dict(n_routed=n_v)))
        hv = self.vision.glm_H(L, E, alpha=alpha, ctx_mass=ctx_mass, device=device)
        H = [((1 - w) * _nt(a.double()) + w * _nt(b.double())).float() for a, b in zip(ht["H"], hv["H"])]
        G = []
        for a, b in zip(ht["G"], hv["G"]):
            if a is None:
                G.append(None); continue
            ga, gb = torch.diagonal(a).double() ** 2, torch.diagonal(b).double() ** 2
            G.append(torch.diag(((1 - w) * ga / ga.mean() + w * gb / gb.mean()).clamp_min(1e-30).sqrt().float()))
        meta = dict(ht["meta"], w_vision=self.w, w_vision_eff=w, vision=dict(n_routed=hv["meta"]["n_routed"], n_ctx=hv["meta"]["n_ctx"],
                                                        ess=hv["meta"]["ess"], cp2=hv["meta"]["cp2"]))
        return dict(H=H, G=G, meta=meta)

    def salience_vision(self, L):
        return self.vision.salience(L)

    def fixed_set_score(self, L, text_weights=None):
        """(1 - w) S_text / sum S_text + w S_vis / sum S_vis per expert, S = sum p ||y|| (token-weighted REAP, sal col 4).
        text_weights = the boundary weight spec of the text fixed set (fixed_set19: think/end d1 50, d2_4 20, d5_16 5,
        d17_32 2); the vision side has no boundary rows (weight 1)."""
        if text_weights is None:
            st = self.text.salience(L)["sums"][:, 0, 4]
        else:
            st = self.text.salience(L, weights=text_weights)["sums"][:, 4]
        sv = self.vision.salience(L)["sums"][:, 0, 4]
        return (1 - self.w) * st / st.sum() + self.w * sv / max(sv.sum(), 1e-300)
