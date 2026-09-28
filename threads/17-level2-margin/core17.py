"""Thread 17 core: data, thread-08 Hessian, generalized nested EXL3-frame quantizer (dual feedback states,
optional two-sided output metric G, pluggable base tile quantizer), holdout + capture scoring."""
import os, sys, math, json, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "5")
sys.path.insert(0, "/home/coder/git/nestquant/threads/05-exl3-harness"); sys.path.insert(0, "/home/coder/git/orbit-duet")
import torch, torch.nn.functional as F
import harness as h
h.gpu_cap(12)
Qm = h._ex()
OD = "/home/coder/git/orbit-duet"
SCR = "/tmp/nestquant/17-level2-margin"
os.makedirs(SCR, exist_ok=True)
EXT = h.ExtTileQuantizer("mul1")


def T0(): return time.time()


# ------------------------------------------------------------------ data
class Exp:
    def __init__(self, L, E, split="full", capture=True):
        self.L, self.E = L, E
        self.data = h.load_expert(L, E, capture="matched" if capture else None)
        self.W = self.data.teacher
        ts = torch.load(f"{OD}/runs/glm53_pilot_matched_l{L}/statistics/l{L}_e{E}_training_sample.pt",
                        map_location="cpu", mmap=True, weights_only=False)
        self.x, self.p = ts["x"], ts["p"].float()
        n = len(self.x); perm = torch.randperm(n, generator=torch.Generator().manual_seed(606))
        self.val = perm[:n // 5].sort().values
        self.fit = perm[n // 5:].sort().values if split == "holdout" else torch.arange(n)
        self.split = split
        self._grams = None

    @torch.no_grad()
    def grams(self):
        """p^2-weighted and uniform input grams (gate/up: x, down: teacher hidden) + SwiGLU output-side energies."""
        if self._grams is not None: return self._grams
        g, u, d = self.W
        Z = lambda k: torch.zeros(k, k, device="cuda", dtype=torch.float64)
        o = {k: Z(6144) for k in ["Xw", "Xu"]}; o.update({k: Z(2048) for k in ["Aw", "Au"]})
        cg = {k: torch.zeros(2048, device="cuda", dtype=torch.float64) for k in ["gw", "gu", "uw", "uu"]}
        for i in range(0, len(self.fit), 2048):
            j = self.fit[i:i + 2048]; x = self.x[j].cuda().float(); p = self.p[j].cuda().float()[:, None]
            gx, ux = x @ g.T, x @ u.T; sig = gx.sigmoid()
            a = (F.silu(F.linear(x.bfloat16(), g.bfloat16())) * F.linear(x.bfloat16(), u.bfloat16())).float()
            dg = ux * sig * (1 + gx * (1 - sig)); du = F.silu(gx)
            # thread 06 'GOdiag': per hidden channel energy of c (no W_d factor)
            for key, wt in [("w", p), ("u", torch.ones_like(p))]:
                xw = (x * wt).double(); o["X" + key].addmm_(xw.T, xw)
                aw = (a * wt).double(); o["A" + key].addmm_(aw.T, aw)
                cg["g" + key] += ((dg * wt).double() ** 2).sum(0); cg["u" + key] += ((du * wt).double() ** 2).sum(0)
        o.update(cg)
        self._grams = o
        return o

    def H(self, proj, alpha=0.75):
        o = self.grams()
        nt = lambda A: A / A.diagonal().mean()
        k = "X" if proj < 2 else "A"
        return ((1 - alpha) * nt(o[k + "w"]) + alpha * nt(o[k + "u"])).float()

    def Gdiag(self, proj, alpha=0.75):
        """thread 06 output metric diag(O), O = sum w c^2 (gate: silu'(g)u, up: silu(g)), mixed like H."""
        o = self.grams(); k = "g" if proj == 0 else "u"
        a, b = o[k + "w"], o[k + "u"]
        return ((1 - alpha) * a / a.mean() + alpha * b / b.mean()).float()

    # ---------------- scoring
    @torch.no_grad()
    def holdout(self, methods):
        """p^2-weighted rel. output L2 (%) on held-out training rows (only meaningful for split=holdout)."""
        acc = {k: [0., 0.] for k in methods}
        tb = [w.bfloat16() for w in self.W]
        for a in range(0, len(self.val), 1024):
            j = self.val[a:a + 1024]; x = self.x[j].cuda(); p2 = self.p[j].cuda().double().square()
            y = h._teacher(x, tb).double(); den = float((y.square().sum(-1) * p2).sum())
            for k, w in methods.items():
                e = (h._teacher(x, [t.bfloat16() for t in w]).double() - y).square().sum(-1)
                acc[k][0] += float((e * p2).sum()); acc[k][1] += den
        return {k: 100 * (v[0] / v[1]) ** .5 for k, v in acc.items()}

    def capture(self, methods):
        tab = h.table(h.evaluate(self.data, methods))
        return {m: {"routed": v["all/routed"], "forced": v["all/forced"], "ood": v["ood/forced"], "ood_r": v.get("ood/routed")}
                for m, v in tab.items()}


SIG = (0.5, 0.5, 1.0)


# ------------------------------------------------------------------ quantizer
def _block_ldl(Hm, sigma):
    Lf, Hr = Qm.block_ldl(Hm, 16, {"sigma_reg": sigma}, False)
    Lf.diagonal().zero_()
    return Lf, Hr.to('cuda')


@torch.no_grad()
def quantize(W, H, *, K=2, sigma=0.5, seed=91426, base_q=None, lam=0.0, level4=False, G=None, sigma_out=None,
             res_gain=1.0, gscale_mult=1.0, refit=True, prep_hook=None, post_hook=None, want_idx=False, gs_fixed=None,
             frozen_base=None, cd=None):
    """EXL3-frame quantization (rotation, regularize, g-scale, block-16 LDL, refit) with
    - base_q: callable(tiles[T,256], K) -> (q, idx) for the base code (default mul1 CUDA Viterbi)
    - level4: nested K2 mul1 residual plane with per-16x16-tile fp16 delta, separate E2/E4 feedback states
    - lam: base target (1-lam) t2 + lam t4   (thread 02 blend)
    - G: output-side metric (vector = diagonal, original output basis) -> two-sided LDLQ (ldlq_2hess ordering)
    Returns dict(W2, W4, info)."""
    dev = torch.device("cuda")
    base_q = base_q or EXT
    weight = W.to(dev, torch.float32).T.contiguous()          # (k=in, n=out)
    k, n = weight.shape
    torch.manual_seed(seed)
    Hm = H.to(dev, torch.float32).clone()
    diag_mean = torch.diag(Hm).mean().item()
    Hm.diagonal().add_(sigma * diag_mean)
    H_diag = Hm.diagonal().clone()
    su = (torch.randn(k, device=dev).sign() + 1e-5).sign().float().unsqueeze(1)
    su_signs = su.clone()
    Hm *= su.T; Qm.blockwise_preapply_had_r_(Hm, 128); Hm *= su; Qm.blockwise_preapply_had_l_(Hm, 128)
    Lf, Hr = _block_ldl(Hm, sigma)
    del Hm
    sv = (torch.randn(n, device=dev).sign() + 1e-5).sign().float().unsqueeze(0)
    weight_orig = weight.clone()
    d = torch.sort(H_diag.sqrt(), descending=True).values
    aos = (d[:k // 50].sum() / d.sum()).item() < 0.15
    ocs = Qm.block_rms(weight, dim=0, keepdim=True); ocs /= ocs.mean().item()
    zero = ocs.abs() < 1e-30
    if aos:
        ocs[zero] = 0.1; sv = (sv * ocs + 1e-10).float()
    weight /= sv; sv[zero] = 0.0
    Qm.blockwise_preapply_had_r_(weight, 128)
    ics = Qm.block_rms(weight, dim=1, keepdim=True); ics[ics.abs() < 1e-30] = 0.1
    su = (su * ics / (-Qm.codebook_scale) + 1e-10).float()
    weight /= su
    Qm.blockwise_preapply_had_l_(weight, 128)
    if gs_fixed is not None:
        gs = gs_fixed
    else:
        tiles = Qm.sample_scale_tiles(weight, 3) * Qm.ldlq_drift(K)
        gs, _ = h._g_scale_search(tiles, K, base_q)
    gs *= gscale_mult
    weight *= gs; su /= gs
    if prep_hook is not None:
        prep_hook(weight=weight, Hr=Hr, Lf=Lf)

    # ---- output metric
    Ln = None
    if G is not None:
        Gm = torch.diag(G.to(dev).float()) if G.dim() == 1 else G.to(dev).float()
        L_out, _ = Qm.prepare_H_out(Gm, sv, {"sigma_reg": sigma, "sigma_reg_out": sigma if sigma_out is None else sigma_out}, False, dev)
        Ln = L_out.clone(); Ln.diagonal().fill_(1.0)
        del Gm, L_out
    Lk = Lf.clone(); Lk.diagonal().fill_(1.0); del Lf

    tk, tn = k // 16, n // 16
    perm, perm_i = Qm.tensor_core_perm(dev), Qm.tensor_core_perm_i(dev)
    ar16 = torch.arange(16, device=dev)
    W4v = weight.view(tk, 16, tn, 16)
    Q2 = torch.zeros_like(weight); Q4 = torch.zeros_like(weight) if level4 else None
    F2 = torch.zeros_like(weight); F4 = torch.zeros_like(weight) if (level4 or lam) else None
    Q2v = Q2.view(tk, 16, tn, 16); Q4v = Q4.view(tk, 16, tn, 16) if level4 else None
    F2v = F2.view(tk, 16, tn, 16); F4v = F4.view(tk, 16, tn, 16) if F4 is not None else None
    enc = torch.zeros((tk, tn, 256), dtype=torch.int16, device=dev) if want_idx else None
    deltas = torch.zeros((tk, tn), device=dev) if level4 else None
    stream = Qm.get_quant_stream(dev); torch.cuda.synchronize(); stream.wait_stream(torch.cuda.current_stream())

    def upd(Fm, a, c, dE):
        rows = (a.unsqueeze(1) * 16 + ar16).flatten()
        k_hi = int(a.max()) * 16 + 16; n_hi = int(c.max()) * 16 + 16
        if Ln is None:
            # one-sided: a is one strip (all tile columns): F[:k_hi] += Lk[strip rows]^T dE_strip
            S = dE.permute(1, 0, 2).reshape(16, -1)
            Fm[:k_hi].addmm_(Lk[rows[:16], :k_hi].T, S)
        else:
            cols = (c.unsqueeze(1) * 16 + ar16)
            right = torch.bmm(dE, Ln[cols][:, :, :n_hi])
            Fm[:k_hi, :n_hi].addmm_(Lk[rows, :k_hi].T, right.reshape(-1, n_hi))

    def tq(t, q=base_q, Kq=K):
        qw, qi = q(t[:, perm].contiguous(), Kq)
        return qw[:, perm_i].view(-1, 16, 16), qi

    with torch.cuda.stream(stream):
        if Ln is None:
            # one-sided: process 16-row strips (all tile columns at once) from the last strip; identical math
            order = [(torch.full((tn,), a, device=dev, dtype=torch.long), torch.arange(tn, device=dev)) for a in range(tk - 1, -1, -1)]
        else:
            order = []
            for s in range(tk + tn - 2, -1, -1):
                a = torch.arange(max(0, s - (tn - 1)), min(tk - 1, s) + 1, device=dev); order.append((a, s - a))
        for a, c in order:
            w = W4v[a, :, c, :]
            t2 = w + F2v[a, :, c, :]
            t4 = w + F4v[a, :, c, :] if F4v is not None else t2
            tb = t2 if not lam else (1 - lam) * t2 + lam * t4
            if frozen_base is not None:
                q2, qi = frozen_base.view(tk, 16, tn, 16)[a, :, c, :], None
            else:
                q2, qi = tq(tb.reshape(-1, 256))
            Q2v[a, :, c, :] = q2
            if enc is not None: enc[a, c] = qi
            if level4:
                r = t4 - q2
                rs = r.square().mean((1, 2)).sqrt().clamp_min(1e-8).half().float()      # fp16 delta per tile
                dq, _ = tq((r * (res_gain / rs)[:, None, None]).reshape(-1, 256), EXT, 2)
                q4 = q2 + dq * (rs / res_gain)[:, None, None]
                Q4v[a, :, c, :] = q4; deltas[a, c] = rs
            if Ln is None and int(a[0]) == 0:
                continue
            upd(F2, a, c, w - q2)
            if F4 is not None:
                upd(F4, a, c, w - (q4 if level4 else q2))
    torch.cuda.current_stream().wait_stream(stream); torch.cuda.synchronize()
    del F2, F4, Lk, Ln
    info = dict(g_scale=gs, aos=aos)
    if cd is not None:
        info["cd"] = cd(weight=weight, Hr=Hr, Q=Q2, tq=tq)
    E = weight - Q2
    info["proxy2"] = Qm.block_trace(E, Hr) / max(Qm.block_trace(weight, Hr), 1e-8)
    if level4:
        E = weight - Q4; info["proxy4"] = Qm.block_trace(E, Hr) / max(Qm.block_trace(weight, Hr), 1e-8)
    del E
    if post_hook is not None:
        post_hook(weight=weight, Hr=Hr, Q2=Q2, Q4=Q4, info=info)
    H_orig = Qm.unrotate_H(Hr.cpu(), su_signs.cpu()) if refit else None

    def decode(Q):
        Wr = Q.clone(); Wr = Qm.preapply_had_l(Wr, 128); Wr *= su; Wr = Qm.preapply_had_r(Wr, 128); Wr *= sv
        s_u, s_v = su, sv
        if refit:
            _, s_u, s_v, _, _ = Qm.refit_scales(weight_orig, Wr, H_orig, su, sv)
            s_u = s_u.view(-1, 1); s_v = s_v.view(1, -1)
        suh, svh = s_u.flatten().half(), s_v.flatten().half()
        wq = Q.half(); wq = Qm.preapply_had_l(wq, 128); wq *= suh.unsqueeze(1); wq = Qm.preapply_had_r(wq, 128); wq *= svh.unsqueeze(0)
        return wq.float().T.contiguous()
    out = dict(W2=decode(Q2), info=info)
    if level4: out["W4"] = decode(Q4)
    if enc is not None: out["idx"] = enc
    if want_idx == "all":
        out.update(Q2r=Q2, weight_r=weight, Hr=Hr, gs=gs)
    del H_orig
    torch.cuda.empty_cache()
    return out


def fit_expert(X, *, sig=SIG, G_beta=None, alpha=0.75, **kw):
    """All three projections. G_beta: use two-sided G = diag(O)^beta on gate/up."""
    W2, W4, infos = [], [], []
    for p in range(3):
        G = X.Gdiag(p, alpha).pow(G_beta) if (G_beta and p < 2) else None
        o = quantize(X.W[p], X.H(p, alpha), sigma=sig[p], G=G, **kw)
        W2.append(o["W2"]); infos.append(o["info"])
        if "W4" in o: W4.append(o["W4"])
        del o; torch.cuda.empty_cache()
    return dict(W2=W2, W4=W4 or None, info=infos)


def exl3_anchor(X, K, sig=SIG, alpha=0.75, **kw):
    out = []
    for p in range(3):
        Wq, _ = h.quantize_exl3_like(X.W[p], X.H(p, alpha), K, count=1, sigma_reg=sig[p], **kw)
        out.append(Wq); torch.cuda.empty_cache()
    h.free_scratch()
    return out


def jdump(obj, path):
    json.dump(obj, open(path, "w"), indent=1, default=float)


# ------------------------------------------------------------------ candidate 3: block coordinate descent on trellis tiles
def make_cd(sweeps=2, order="rev", accept=True, damp_extra=0.0, log=None):
    """After LDLQ: for each 16-row block b, target t_b = W_b - H_bb^-1 sum_{c!=b} H_bc E_c (E = Q - W), re-run the
    tile Viterbi on t_b (MSE), keep each 16x16 tile's new path only if its exact H_bb loss drops."""
    @torch.no_grad()
    def cd(weight, Hr, Q, tq):
        k, n = weight.shape; tk, tn = k // 16, n // 16
        Hd = Hr.float()
        hist = []
        E = Q - weight
        def total(): return float((E * (Hd @ E)).sum() / (weight * (Hd @ weight)).sum())
        hist.append(total())
        for sw in range(sweeps):
            blocks = range(tk - 1, -1, -1) if (order == "rev") ^ (sw % 2 == 1 and order == "alt") else range(tk)
            nacc = 0
            for b in blocks:
                r0, r1 = b * 16, b * 16 + 16
                Hbb = Hd[r0:r1, r0:r1]
                R = Hd[r0:r1] @ E - Hbb @ E[r0:r1]
                shift = torch.linalg.solve(Hbb, R)
                t = weight[r0:r1] - shift                               # (16, n)
                tiles = t.reshape(16, tn, 16).permute(1, 0, 2)          # (tn, 16, 16)
                qn, _ = tq(tiles.reshape(-1, 256))
                qn = qn.permute(1, 0, 2).reshape(16, n)
                # exact loss of block b given the rest: (e + shift)^T Hbb (e + shift), e = q - w
                def bl(qq):
                    z = qq - weight[r0:r1] + shift
                    return (z * (Hbb @ z)).sum(0).view(tn, 16).sum(1)
                lo, ln = bl(Q[r0:r1]), bl(qn)
                m = (ln < lo) if accept else torch.ones_like(lo, dtype=torch.bool)
                mm = m.repeat_interleave(16)[None]
                Q[r0:r1] = torch.where(mm, qn, Q[r0:r1]); nacc += int(m.sum())
                E[r0:r1] = Q[r0:r1] - weight[r0:r1]
            hist.append(total())
            if log: log(f"cd sweep {sw} proxy {hist[-1]:.6f} accepted {nacc}/{tk*tn}")
        return hist
    return cd


# ------------------------------------------------------------------ sequential down (fit down against quantized hidden)
@torch.no_grad()
def hq_grams(X, gq, uq, alpha=0.75):
    """Mixed (0.25 p^2 + 0.75 uniform) grams of quantized hidden Hq, cross C = sum h hq^T, teacher A (same normalisation)."""
    g, u, d = X.W
    Z = lambda: torch.zeros(2048, 2048, device="cuda", dtype=torch.float64)
    o = {f"{k}{w}": Z() for k in "AQC" for w in "wu"}
    gb, ub, gqb, uqb = g.bfloat16(), u.bfloat16(), gq.bfloat16(), uq.bfloat16()
    for i in range(0, len(X.fit), 2048):
        j = X.fit[i:i + 2048]; x = X.x[j].cuda(); p = X.p[j].cuda().float()[:, None]
        a = (F.silu(F.linear(x, gb)) * F.linear(x, ub)).float(); aq = (F.silu(F.linear(x, gqb)) * F.linear(x, uqb)).float()
        for w, wt in [("w", p), ("u", torch.ones_like(p))]:
            A_, Q_ = (a * wt).double(), (aq * wt).double()
            o["A" + w].addmm_(A_.T, A_); o["Q" + w].addmm_(Q_.T, Q_); o["C" + w].addmm_(A_.T, Q_)
    sw, su_ = o["Aw"].diagonal().mean(), o["Au"].diagonal().mean()
    mix = lambda k: ((1 - alpha) * o[k + "w"] / sw + alpha * o[k + "u"] / su_)
    return mix("A"), mix("Q"), mix("C")


def seq_down_target(X, gq, uq, ridge=1.0, alpha=0.75):
    """W* = argmin sum ||W_d h - W hq||^2 + ridge*mean(diag Hq)*||W - W_d||^2  = W_d (C + r I)(Hq + r I)^-1."""
    A, Hq, C = hq_grams(X, gq, uq, alpha)
    lam = ridge * float(Hq.diagonal().mean())
    I = torch.eye(2048, device="cuda", dtype=torch.float64)
    Ws = (X.W[2].double() @ (C + lam * I) @ torch.linalg.inv(Hq + lam * I)).float()
    return Ws, Hq.float(), A.float()


# ------------------------------------------------------------------ candidate 2: per-tile affine codebook variants (free at decode:
# the variant's (a, b) fold into mul1's final hfma constants, v = h*(a*k_inv) + a*k_bias + b)
class VariantQ:
    def __init__(self, variants, inner=EXT):
        self.variants, self.inner, self.stats = variants, inner, torch.zeros(len(variants))
    def __call__(self, tiles, K):
        best = bi = None
        for vi, (a, b) in enumerate(self.variants):
            q, i = self.inner((tiles - b) / a, K)
            q = q * a + b
            m = (q - tiles).square().sum(1)
            if best is None:
                best, bq, bidx, sel = m, q, i, torch.zeros_like(m, dtype=torch.long)
            else:
                t = m < best
                best = torch.where(t, m, best); bq = torch.where(t[:, None], q, bq); bidx = torch.where(t[:, None], i, bidx)
                sel = torch.where(t, vi, sel)
        self.stats += torch.bincount(sel.cpu(), minlength=len(self.variants)).float()
        return bq, bidx
    @property
    def bits(self): return math.log2(len(self.variants))

VARIANTS = {"sign": [(1.0, 0.0), (-1.0, 0.0)],
            "sign_gain2": [(0.95, 0.0), (1.05, 0.0), (-0.95, 0.0), (-1.05, 0.0)],
            "sign_gain4": [(s * g, 0.0) for g in (0.9, 0.97, 1.03, 1.1) for s in (1, -1)]}
