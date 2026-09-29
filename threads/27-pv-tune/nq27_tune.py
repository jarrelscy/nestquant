"""T27: PV-style post-tuning of the continuous parameters of one NestQuant expert artifact (no format change).

Trainable (all stored fp16 in the artifact; bytes layout, shapes, dtypes and codes unchanged):
  per projection p in gate/up/down and level l in {2 (base plane), 4 (p4 plane)}:  suh_l [in], svh_l [out]
  low-rank plane (if r > 0): V [r, in] (gate/up share one V: meta shared_V), U2 [r, out], U4 [r, out]
Model of the decode (nq_decode.dense_from_rotated + apply_ocol, in fp32):
  W_{p,l} = (su_l[:, None] * B_{p,l} * sv_l[None, :])^T + U2^T V (+ U4^T V at l = 4),  B = Had_k Q_l Had_n
  (Q_l = the frozen trellis levels of the artifact in the rotated basis).  The decoder rounds to fp16 between the
  steps; the tuned artifact is always re-scored through nq_decode itself.
Objectives (loss = a * err(L2) + (1 - a) * err(L4), err = relative squared error):
  act  per-expert router-weighted SwiGLU output error on real routed fit rows (T19 chunk-0 activations, nq27_rows),
       i.e. the harness 'routed' metric: sum p^2 ||y_q(x) - y_T(x)||^2 / sum p^2 ||y_T||^2 (teacher = harness bf16).
  H    per-projection proxy tr(G E H E^T G) / tr(G W H W^T G) on the full 15.4M-token stats (routed p^2 Grams,
       undamped; G = routed diag output weights on gate/up, I on down), summed over the 3 projections.
Early stopping / selection: held-out 2048-token blocks of the chunk-0 rows (blk % 10 == 0, never in a gradient),
scored with the act metric for every objective.  eval/val (the spot rows) is never touched here.
"""
import os, sys, copy, math, time, json
import torch
import torch.nn.functional as F

import nq_decode as D

PROJ = ("gate", "up", "down")
DECODE_DEV = os.environ.get("NQ27_DECODE_DEV", "cuda")      # nq_decode ring expansion needs ~2-3 GB on GPU


def had(n, dev):
    from exllamav3.modules.quant.exl3_lib import quantize as Qm
    return Qm.get_hadamard_dt(n, dev, torch.float32, 1 / math.sqrt(n))


def rotB(Q, dev):
    """Had_k Q Had_n in fp32 (k-blocks of 128 on the left, n-blocks on the right), [k, n]."""
    Hh = had(128, dev)
    k, n = Q.shape
    X = (Hh @ Q.float().view(-1, 128, n)).view(k, n)
    return (X.view(k, -1, 128) @ Hh).view(k, n)


class ExpertParams(torch.nn.Module):
    def __init__(self, art, dev, tune=("su", "sv", "U", "V")):
        super().__init__()
        self.dev = dev
        self.B, self.s0, self.lr0 = {}, {}, {}
        self.tune = set(tune)
        for p in PROJ:
            P = art[p]
            rot = {l: q.to(dev) for l, q in D.rotated_levels(P, DECODE_DEV).items()}
            for l in (2, 4):
                pl = P[D.SCALE_PLANE[l]]
                self.B[p, l] = rotB(rot[l], dev)
                self.s0[p, l] = (pl["suh"].to(dev).float(), pl["svh"].to(dev).float())
                self.register_parameter(f"a_{p}{l}", torch.nn.Parameter(torch.zeros_like(self.s0[p, l][0]),
                                                                        requires_grad="su" in self.tune))
                self.register_parameter(f"b_{p}{l}", torch.nn.Parameter(torch.zeros_like(self.s0[p, l][1]),
                                                                        requires_grad="sv" in self.tune))
            del rot
            lr = P["base"].get("lr")
            if lr is not None and lr["V"].shape[0] > 0:
                V, U2, U4 = lr["V"].to(dev).float(), lr["U2"].to(dev).float(), P["p4"]["lr"]["U4"].to(dev).float()
                self.lr0[p] = dict(V=V, U2=U2, U4=U4, sV=V.square().mean(1, keepdim=True).sqrt().clamp_min(1e-8),
                                   sU2=U2.square().mean(1, keepdim=True).sqrt().clamp_min(1e-8),
                                   sU4=U4.square().mean(1, keepdim=True).sqrt().clamp_min(1e-8))
                self.register_parameter(f"u2_{p}", torch.nn.Parameter(torch.zeros_like(U2), requires_grad="U" in self.tune))
                self.register_parameter(f"u4_{p}", torch.nn.Parameter(torch.zeros_like(U4), requires_grad="U" in self.tune))
                if not (p == "up" and P["meta"].get("lr", {}).get("shared_V")):
                    self.register_parameter(f"v_{p}", torch.nn.Parameter(torch.zeros_like(V), requires_grad="V" in self.tune))
        self.shared_V = bool(art["up"]["meta"].get("lr", {}).get("shared_V"))

    def scales(self, p, l):
        su0, sv0 = self.s0[p, l]
        return su0 * (1 + getattr(self, f"a_{p}{l}")), sv0 * (1 + getattr(self, f"b_{p}{l}"))

    def lr(self, p):
        if p not in self.lr0:
            return None
        z = self.lr0[p]
        vp = "gate" if (p == "up" and self.shared_V) else p
        V = self.lr0[vp]["V"] + self.lr0[vp]["sV"] * getattr(self, f"v_{vp}")
        U2 = z["U2"] + z["sU2"] * getattr(self, f"u2_{p}")
        U4 = z["U4"] + z["sU4"] * getattr(self, f"u4_{p}")
        return V, U2, U4

    def W(self, p, l):
        su, sv = self.scales(p, l)
        W = (su[:, None] * self.B[p, l] * sv[None, :]).T
        z = self.lr(p)
        if z is not None:
            V, U2, U4 = z
            W = W + U2.T @ V
            if l == 4:
                W = W + U4.T @ V
        return W

    def weights(self, l):
        return [self.W(p, l) for p in PROJ]

    @torch.no_grad()
    def write(self, art):
        """-> new artifact: same object tree, only suh/svh/U2/U4/V values replaced (fp16)."""
        new = copy.deepcopy(art)
        for p in PROJ:
            for l in (2, 4):
                su, sv = self.scales(p, l)
                pl = new[p][D.SCALE_PLANE[l]]
                pl["suh"] = su.half().cpu(); pl["svh"] = sv.half().cpu()
            z = self.lr(p)
            if z is not None:
                V, U2, U4 = z
                new[p]["base"]["lr"]["V"] = V.half().cpu()
                new[p]["base"]["lr"]["U2"] = U2.half().cpu()
                new[p]["p4"]["lr"]["U4"] = U4.half().cpu()
        return new


def swiglu(x, W):
    g, u, d = W
    return F.linear(F.silu(F.linear(x, g)) * F.linear(x, u), d)


def teacher_bf16(x, tb):
    g, u, d = tb
    xb = x.bfloat16()
    return F.linear(F.silu(F.linear(xb, g)) * F.linear(xb, u), d).float()


def layout_check(a, b, path=""):
    """Everything except the tuned fp16 value tensors must be identical; tuned ones keep shape/dtype."""
    TUNED = ("suh", "svh", "V", "U2", "U4")
    if isinstance(a, dict):
        assert set(a) == set(b), (path, set(a) ^ set(b))
        for k in a:
            layout_check(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            layout_check(x, y, f"{path}[{i}]")
    elif torch.is_tensor(a):
        assert a.dtype == b.dtype and a.shape == b.shape, path
        if path.split("/")[-1] not in TUNED:
            assert torch.equal(a, b), path
    else:
        assert a == b, path


class ActData:
    """Routed fit rows of one expert on the GPU (bf16), train / held-out split by 2048-token block."""
    def __init__(self, rows_pt, dev, hold_mod=10, max_train=None, seed=0, train_dev=None):
        r = torch.load(rows_pt, weights_only=True)
        hold = (r["blk"] % hold_mod) == 0
        self.xt, self.pt = r["x"][~hold], r["p"][~hold]
        self.xh, self.ph = r["x"][hold].to(dev), r["p"][hold].to(dev)
        if max_train and len(self.pt) > max_train:
            g = torch.Generator().manual_seed(seed)
            i = torch.randperm(len(self.pt), generator=g)[:max_train]
            self.xt, self.pt = self.xt[i], self.pt[i]
        self.xt = self.xt.to(train_dev or dev); self.pt = self.pt.to(dev)
        if self.xt.device.type == "cpu":
            self.xt = self.xt.pin_memory()
        self.n_train, self.n_hold = len(self.pt), len(self.ph)
        self.ot = self.oh = None

    @classmethod
    def from_tensors(cls, x, p, blk, dev, off=None, hold_mod=10, train_dev=None):
        """Same split rule on in-memory rows; off [N, out] (bf16) = additive target offset (band 'ref' target)."""
        self = cls.__new__(cls)
        hold = (blk % hold_mod) == 0
        self.xt, self.pt = x[~hold].to(train_dev or dev), p[~hold].to(dev)
        self.xh, self.ph = x[hold].to(dev), p[hold].to(dev)
        self.ot = off[~hold].to(train_dev or dev) if off is not None else None
        self.oh = off[hold].to(dev) if off is not None else None
        self.n_train, self.n_hold = len(self.pt), len(self.ph)
        return self


@torch.no_grad()
def act_err(x, p, tb, Ws, batch=4096, off=None):
    """{name: relative router-weighted L2 (not squared)} for several weight sets on rows x (target = teacher + off)."""
    num = {k: 0. for k in Ws}; den = 0.
    for i in range(0, len(p), batch):
        xb = x[i:i + batch]; p2 = p[i:i + batch].double().square()
        yT = teacher_bf16(xb, tb).double()
        if off is not None:
            yT = yT + off[i:i + batch].double()
        den += float((yT.square().sum(-1) * p2).sum())
        xf = xb.float()
        for k, W in Ws.items():
            num[k] += float(((swiglu(xf, W).double() - yT).square().sum(-1) * p2).sum())
    return {k: (v / den) ** .5 for k, v in num.items()}


def load_H(cap, L, E, dev):
    """Routed p^2-weighted, undamped proxy metrics from the full T19 stats: H_x = A2/sum p^2, H_h = D2/sum p^2,
    G_gate = diag(g0 / sum p^2)^1/2, G_up = diag(g2 / sum p^2)^1/2 (per-output-channel SwiGLU output weight)."""
    c = cap.components(L, E, dev, keys=("A2", "D2"))
    s = c["sum_p2"]
    Hx = c["A2"] / s; Hh = c["D2"] / s
    g = c["g"].float()
    return {"gate": (Hx, (g[0] / s).clamp_min(0)), "up": (Hx, (g[2] / s).clamp_min(0)), "down": (Hh, None)}


def H_loss(Wq, WT, Hm):
    tot = 0.
    for p, Wp, Tp in zip(PROJ, Wq, WT):
        H, g = Hm[p]
        E = Wp - Tp
        EH = E @ H; TH = Tp @ H
        if g is not None:
            num = ((EH * E).sum(1) * g).sum(); den = ((TH * Tp).sum(1) * g).sum()
        else:
            num = (EH * E).sum(); den = (TH * Tp).sum()
        tot = tot + num / den
    return tot / 3


def tune(art, teacher, data, *, objective="act", Hm=None, a=0.5, steps=400, lr=1e-3, batch=4096, eval_every=25,
         patience=6, tune_set=("su", "sv", "U", "V"), dev="cuda", log=print, seed=0, lr_decay=True, norm=False, warmup=0, heavy=0, cap_q=0., gamma=0., fdata=None, fbatch=1024):
    """norm=True: level-balanced loss a * err2/err2_0 + (1 - a) * err4/err4_0 (err_0 = the untuned value; held-out
    initial for act/selection, initial H proxy for H) -- without it L2 (~5x larger rel. error) dominates ~20:1."""
    torch.manual_seed(seed)
    M = ExpertParams(art, dev, tune_set)
    tb = [t.to(dev).bfloat16() for t in teacher]
    tf = [t.to(dev).float() for t in teacher]
    params = [q for q in M.parameters() if q.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    wu = max(int(warmup), 0)
    fn = lambda s: min(1., (s + 1) / wu) if wu else 1.
    if lr_decay:
        fn = (lambda f: lambda s: f(s) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))(fn)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, fn)
    # global denominator (router-weighted teacher energy) on the train rows
    en = []
    with torch.no_grad():
        for i in range(0, data.n_train, batch):
            yT = teacher_bf16(data.xt[i:i + batch].to(dev), tb)
            if data.ot is not None:
                yT = yT + data.ot[i:i + batch].to(dev).float()
            en.append(yT.double().square().sum(-1) * data.pt[i:i + batch].double().square())
    en = torch.cat(en); den = float(en.sum())
    den_b = den / data.n_train
    # stratified batches: the `heavy` highest-energy train rows (a handful of tokens can carry >50% of an expert's
    # p^2-weighted output energy) are in every batch with weight 1; the rest are sampled with weight n_rest/b_rest.
    hv = min(int(heavy), batch // 2, data.n_train // 4) if heavy else 0
    if hv:
        hidx = en.topk(hv).indices
        rest = torch.ones(data.n_train, dtype=torch.bool, device=en.device); rest[hidx] = False
        ridx = rest.nonzero().squeeze(1)
        w_rest = len(ridx) / (batch - hv)
        wts = torch.cat([torch.ones(hv, device=dev), torch.full((batch - hv,), w_rest, device=dev)]).float()
        hshare = float(en[hidx].sum()) / den
    else:
        hshare = 0.
    ess = float(den ** 2 / float(en.square().sum()))      # effective sample size of the energy weights
    # cap_q: robust train loss -- each row's weight is capped at the cap_q energy quantile (min(1, q / e_i))
    cw = (en.float().quantile(cap_q) / en.float()).clamp(max=1.).float() if cap_q else None

    # forced regularizer (gamma > 0): error of the expert applied to UNIFORM (not routed) chunk-0 rows, weight 1 --
    # the harness 'forced' metric; keeps channels without routed signal from drifting (routing-drift robustness).
    if gamma:
        fxt, fxh = fdata
        with torch.no_grad():
            fden_b = sum(float(teacher_bf16(fxt[i:i + 4096], tb).double().square().sum())
                         for i in range(0, len(fxt), 4096)) / len(fxt)
        fones = torch.ones(len(fxh), device=dev)
    fn2 = {2: 1., 4: 1.}

    def held():
        with torch.no_grad():
            W = {2: M.weights(2), 4: M.weights(4)}
            e = act_err(data.xh, data.ph, tb, W, off=data.oh)
            j = a * e[2] ** 2 / n2[2] + (1 - a) * e[4] ** 2 / n2[4]
            if gamma:
                ef = act_err(fxh, fones, tb, W)
                e.update({"f2": ef[2], "f4": ef[4]})
                j += gamma * (a * ef[2] ** 2 / fn2[2] + (1 - a) * ef[4] ** 2 / fn2[4])
        return e, j

    n2 = {2: 1., 4: 1.}
    e0, _ = held()
    if norm:
        n2 = {l: e0[l] ** 2 for l in (2, 4)}
        if gamma:
            fn2 = {l: e0[f"f{l}"] ** 2 for l in (2, 4)}
    j0 = held()[1]
    if norm and objective == "H":
        with torch.no_grad():
            nH = {l: float(H_loss(M.weights(l), tf, Hm)) for l in (2, 4)}
    else:
        nH = {2: 1., 4: 1.}
    best = dict(step=0, j=j0, e=e0, state={k: v.detach().clone() for k, v in M.state_dict().items()})
    hist = [dict(step=0, L2=e0[2], L4=e0[4], j=j0)]
    log(f"  step 0 held L2 {100*e0[2]:.3f} L4 {100*e0[4]:.3f}" + (f" F2 {100*e0['f2']:.3f} F4 {100*e0['f4']:.3f}" if gamma else ""))
    bad = 0; t0 = time.time()
    g = torch.Generator(device=dev).manual_seed(seed)
    for step in range(1, steps + 1):
        if objective == "act":
            if hv:
                i = torch.cat([hidx, ridx[torch.randint(0, len(ridx), (batch - hv,), device=dev, generator=g)]])
            else:
                i = torch.randint(0, data.n_train, (batch,), device=dev, generator=g)
            xb = data.xt[i.to(data.xt.device)].to(dev, non_blocking=True); p2 = data.pt[i].square()
            if hv:
                p2 = p2 * wts / (data.n_train / batch)
            if cw is not None:
                p2 = p2 * cw[i]
            with torch.no_grad():
                yT = teacher_bf16(xb, tb)
                if data.ot is not None:
                    yT = yT + data.ot[i.to(data.ot.device)].to(dev, non_blocking=True).float()
            xf = xb.float()
            loss = 0.
            for l, wl in ((2, a), (4, 1 - a)):
                if wl == 0:
                    continue
                e = ((swiglu(xf, M.weights(l)) - yT).square().sum(-1) * p2).mean() / den_b
                loss = loss + wl * e / n2[l]
            if gamma:
                fi = torch.randint(0, len(fxt), (fbatch,), device=dev, generator=g)
                fb = fxt[fi]
                with torch.no_grad():
                    fyT = teacher_bf16(fb, tb)
                for l, wl in ((2, a), (4, 1 - a)):
                    if wl:
                        ef = (swiglu(fb.float(), M.weights(l)) - fyT).square().sum(-1).mean() / fden_b
                        loss = loss + gamma * wl * ef / fn2[l]
        else:
            loss = 0.
            for l, wl in ((2, a), (4, 1 - a)):
                if wl:
                    loss = loss + wl * H_loss(M.weights(l), tf, Hm) / nH[l]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if sched:
            sched.step()
        if step % eval_every == 0 or step == steps:
            e, j = held()
            hist.append(dict(step=step, L2=e[2], L4=e[4], j=j, train=float(loss.detach())))
            if j < best["j"]:
                best = dict(step=step, j=j, e=e, state={k: v.detach().clone() for k, v in M.state_dict().items()})
                bad = 0
            else:
                bad += 1
            log(f"  step {step} loss {float(loss.detach()):.5f} held L2 {100*e[2]:.3f} L4 {100*e[4]:.3f} "
                + (f"F2 {100*e['f2']:.3f} F4 {100*e['f4']:.3f} " if gamma else "") +
                f"best@{best['step']} ({time.time()-t0:.0f}s)")
            if bad >= patience:
                break
    M.load_state_dict(best["state"])
    return M, dict(best_step=best["step"], held0={str(k): v for k, v in e0.items()},
                   held_best={str(k): v for k, v in best["e"].items()}, hist=hist, n_train=data.n_train,
                   n_hold=data.n_hold, time=time.time() - t0, norm=norm, warmup=wu, heavy=hv,
                   heavy_energy_share=hshare, energy_ess=ess, cap_q=cap_q, gamma=gamma)
