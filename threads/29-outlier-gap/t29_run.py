"""T29: NestQuant L2 vs EXL3-2 gap, encoder/EXL3 ingredient arms scored on held-out val (T27 / nq25_spot --eval val).

  python t29_run.py --arms prod,nolr,lam0,X2,X2+intra0 L:E [L:E ...]

Same H as the campaign encode and the T27 EXL3 refs: H = T12 nq_layer.expert_HG(open_stats(_stats, _stats_mm, 0.25)).
Score = harness._errors arithmetic on the val routed rows (= T27 val/base 'all/routed'), plus
  top10: rel. error on the 10 highest p^2||y_T||^2 rows / rest; share of the error on those rows,
  per projection: eH = tr(E H E^T)/tr(W H W^T) with the calibration H, eV = same with the val routed Gram
  (sum p^2 x x^T; down: teacher SwiGLU act), for the calibration vs held-out split of the gap.
Arms (tokens joined by '+'):
  nq:   prod (T12 nq_encode.encode_expert PROD = campaign t23b, bit-identical) | art (decode campaign artifact)
        tokens: nolr, lam<x>, dtau<x>, dr<n> (down-only lr), noG, inner<n>, novar, tau<x>, r<n>, sig<gu>_<d>, hook:<module>:<fn> (fn(Ws, HG, kw) -> Ws, HG, kw)
  EXL3: X2 / X4, tokens intra0, noldlq, aos, noaos, norefit, sig<gu>_<d>, nogs
-> /tmp/nestquant/29-outlier-gap/res/L{L}_E{E}/{arm}.json
"""
import os, sys, json, time, argparse, importlib
os.environ.setdefault("OMP_NUM_THREADS", "8")
import torch

SRC = "/tmp/nestquant/src/glm53-fp8"
R = "/tmp/nestquant/nq-encode-v1"
OUT = "/tmp/nestquant/29-outlier-gap/res"
PROJ = ("gate", "up", "down")


def parse(arm):
    t = arm.split("+")
    if t[0].startswith("X") or t[0] in ("art", "prod"):
        return t[0], t[1:]
    return "prod", t


def sig_tok(tok, sig):
    a, b = tok[3:].split("_")
    return {"gate": float(a), "up": float(a), "down": float(b)}


def exl3_arm(h, Ws, HG, head, toks, sig):
    K = int(head[1:])
    Qm = h._ex()
    kw = dict(count=1)
    s = dict(sig)
    intra0 = False
    for tk in toks:
        if tk == "intra0": intra0 = True
        elif tk == "noldlq": kw["ldlq"] = False
        elif tk == "aos": kw["apply_out_scales"] = True
        elif tk == "noaos": kw["apply_out_scales"] = False
        elif tk == "norefit": kw["refit"] = False
        elif tk == "nogs": kw["g_scale"] = False
        elif tk.startswith("sig"): s = sig_tok(tk, sig)
        else: raise SystemExit(f"bad exl3 token {tk}")
    orig = Qm.block_ldl
    if intra0:
        def patched(Hm, b, *a, **k2):
            L, Hr = orig(Hm, b, *a, **k2)
            for i in range(0, L.shape[0], 128):
                L[i:i + 128, i:i + 128] = 0
            return L, Hr
        Qm.block_ldl = patched
    q, info = [], {}
    try:
        for pi, pn in enumerate(PROJ):
            Wq, inf = h.quantize_exl3_like(Ws[pi], HG["H"][pi], K, sigma_reg=s[pn], **kw)
            q.append(Wq.cpu()); h.free_scratch()
            info[pn] = dict(aos=bool(inf["apply_out_scales"]), skew=float(inf["skew"]), gs=float(inf["g_scale"]),
                            bits=float(inf["bits"]) if not isinstance(inf["bits"], dict) else inf["bits"])
    finally:
        Qm.block_ldl = orig
    return {K: q}, dict(exl3=info)


def nq_arm(NE, D, Ws, HG, head, toks, L, E):
    if head == "art":
        art = torch.load(f"{R}/L{L}/experts/E{E}.pt", weights_only=False, map_location="cpu")
    else:
        kw = dict()
        inter_perm = in_perm = None
        lr = dict(NE.PROD["lr"])
        HGa = dict(HG)
        dlr = {}                                             # down-only low-rank override (dtau<x>, dr<n>)
        for tk in toks:
            if tk == "nolr": lr = None
            elif tk.startswith("lam"): kw["lam"] = float(tk[3:])
            elif tk == "noG": HGa = dict(HGa, G=[None, None, None])
            elif tk.startswith("inner"): kw["inner"] = int(tk[5:])
            elif tk == "novar": kw["base_var"] = None
            elif tk.startswith("dtau"): dlr["tau"] = float(tk[4:])
            elif tk.startswith("dr") and tk[2:].isdigit(): dlr["rmax"] = int(tk[2:])
            elif tk.startswith("tau"): lr["tau"] = float(tk[3:])
            elif tk.startswith("r") and tk[1:].isdigit(): lr["rmax"] = int(tk[1:])
            elif tk.startswith("sig"): kw["sigma"] = sig_tok(tk, NE.PROD["sigma"])
            elif tk.startswith("dperm"):                     # intermediate-channel order (free: gate/up rows + down cols)
                perm = order_of(HGa["H"][2], tk[5:] or "ao", NE.PROD["sigma"]["down"])
                Ws = [Ws[0][perm], Ws[1][perm], Ws[2][:, perm]]
                HGa = dict(HGa, H=[HGa["H"][0], HGa["H"][1], sym_take(HGa["H"][2], perm)],
                           G=[None if g is None else sym_take(g, perm) for g in HGa["G"][:2]] + [HGa["G"][2]])
                inter_perm = perm
            elif tk.startswith("xperm"):                     # hidden-state (gate/up input) order: kernel gathers x
                perm = order_of(HGa["H"][0], tk[5:] or "ao", NE.PROD["sigma"]["gate"])
                Ws = [Ws[0][:, perm], Ws[1][:, perm], Ws[2]]
                Hx = sym_take(HGa["H"][0], perm)
                HGa = dict(HGa, H=[Hx, Hx, HGa["H"][2]])
                in_perm = perm
            elif tk.startswith("hook:"):
                _, mod, fn = tk.split(":")
                Ws, HGa, kw = getattr(importlib.import_module(mod), fn)(Ws, HGa, kw)
                lr = kw.pop("lr", lr)
            else: raise SystemExit(f"bad nq token {tk}")
        kw["lr"] = lr
        orig_detect = NE.lr_detect
        if dlr:
            kd = HGa["H"][2].shape[0]
            assert kd != HGa["H"][0].shape[0]
            def patched(H, **k2):
                return orig_detect(H, **dict(k2, **dlr)) if H.shape[0] == kd else orig_detect(H, **k2)
            NE.lr_detect = patched
        try:
            art, _ = NE.encode_expert(Ws, HGa, **kw)
        finally:
            NE.lr_detect = orig_detect
        if inter_perm is not None:
            art["meta"]["inter_perm"] = inter_perm.tolist()
        if in_perm is not None:
            art["meta"]["t29"] = dict(in_perm=True)
    dense = {Lv: [w.cpu() for w in D.decode_expert(art, Lv)] for Lv in (2, 4)}
    if head != "art" and in_perm is not None:
        inv = torch.argsort(in_perm)
        dense = {Lv: [q[0][:, inv], q[1][:, inv], q[2]] for Lv, q in dense.items()}
    per = {pn: D.bits_per_level(art[pn]) for pn in PROJ}
    nw = {pn: art[pn]["meta"]["k"] * art[pn]["meta"]["n"] for pn in PROJ}
    bpw = {Lv: sum(per[pn][Lv] * nw[pn] for pn in PROJ) / sum(nw.values()) for Lv in (2, 4)}
    meta = dict(bpw=bpw, lr_rank=art["meta"].get("lr_rank"),
                info={pn: {k: v for k, v in art["meta"]["info"][pn].items() if k in ("proxy_rot",)} for pn in PROJ})
    extra = art["meta"].get("t29")
    if extra is not None:
        meta["t29"] = extra
    return dense, meta


def apply_rots(Ws, HG, rtk):
    """drot<kind> / xrot<kind>: extra orthogonal rotation of the down / gate+up input: W' = W R^T, H' = R H R^T."""
    rots = {}
    if not rtk:
        return Ws, HG, rots
    Ws = [w.float() for w in Ws]; HG = dict(HG)
    for tk in rtk:
        pi_ = 2 if tk.startswith("drot") else 0
        Rm = rot_of(HG["H"][pi_].shape[0], tk[4:])
        Rh = Rm.to(HG["H"][pi_].device, HG["H"][pi_].dtype)
        Hn = Rh @ HG["H"][pi_] @ Rh.T
        if pi_ == 2:
            Ws = [Ws[0], Ws[1], Ws[2] @ Rm.T]; HG["H"] = [HG["H"][0], HG["H"][1], Hn]
        else:
            Ws = [Ws[0] @ Rm.T, Ws[1] @ Rm.T, Ws[2]]; HG["H"] = [Hn, Hn, HG["H"][2]]
        rots[pi_] = Rm
    return Ws, HG, rots


def rot_of(k, kind, seed=2929):
    """Orthogonal R [k, k] applied to an input (x' = R x): 'sh' = Sylvester Had of the TP8 shard width (k/8) per shard, '<t>' = Had k/t per TP-t shard,
    'full' = Sylvester Had k (power of 2) , 'q' = dense random orthogonal; each with random signs first."""
    g = torch.Generator().manual_seed(seed)
    sgn = torch.randn(k, generator=g).sign().double()
    def had(n):
        Hm = torch.ones(1, 1, dtype=torch.float64)
        while Hm.shape[0] < n:
            Hm = torch.cat([torch.cat([Hm, Hm], 1), torch.cat([Hm, -Hm], 1)], 0)
        assert Hm.shape[0] == n, n
        return Hm / n ** 0.5
    if kind == "sh":                                  # TP8 shard width (k/8)
        B = torch.block_diag(*[had(k // 8)] * 8)
    elif kind.isdigit():                              # drot4 = TP4 shard width (k/4) blocks
        B = torch.block_diag(*[had(k // int(kind))] * int(kind))
    elif kind == "full":
        B = had(k)
    elif kind == "q":
        B = torch.linalg.qr(torch.randn(k, k, generator=g, dtype=torch.float64))[0]
    else:
        raise SystemExit(f"bad rot {kind}")
    return (B * sgn[None, :]).float()


def sym_take(M, perm):
    p = perm.to(M.device)
    return M[p][:, p].contiguous()


def order_of(H, how, sigma):
    """Input-channel order for the 128-row ring layout (diag29 findings): ao = diag H ascending (largest channels in the
    first-processed = highest-index blocks), pivr = reversed greedy pivoted Cholesky of the damped H."""
    import diag29
    H = H.double().cpu()
    if how == "ao":
        return torch.argsort(H.diagonal())
    if how == "pivr":
        return diag29.pivot_order(H, sigma).flip(0)
    raise SystemExit(f"bad order {how}")


class Scorer:
    def __init__(self, h, dm):
        self.h = h
        cap = dm.capture
        self.rows, slots = torch.where(cap["ids"] == dm.expert)
        self.p = cap["p"][self.rows, slots]
        self.cap = cap
        self.tb = [t.bfloat16() for t in dm.teacher]
        Dm = dm.teacher[0].shape[1]; I = dm.teacher[0].shape[0]
        Gx = torch.zeros(Dm, Dm, device="cuda", dtype=torch.float64)
        Ga = torch.zeros(I, I, device="cuda", dtype=torch.float64)
        en = []
        self.xs = []
        for f in range(0, len(self.rows), 64):
            x = cap["x"][self.rows[f:f + 64]].cuda()
            p = self.p[f:f + 64].cuda().double()
            y = h._teacher(x, self.tb).double()
            en.append(y.square().sum(-1).cpu())
            xb = x.bfloat16()
            a = (torch.nn.functional.silu(torch.nn.functional.linear(xb, self.tb[0])) *
                 torch.nn.functional.linear(xb, self.tb[1])).double()
            xd = x.double() * p[:, None]; ad = a * p[:, None]
            Gx += xd.T @ xd; Ga += ad.T @ ad
        self.energy = torch.cat(en)
        w = self.energy * self.p.double().square()
        self.top = torch.topk(w, min(10, len(w))).indices
        self.Hval = [Gx.float(), Gx.float(), Ga.float()]
        self.w = w

    @torch.no_grad()
    def forced(self, Wq):
        """= harness._errors(cap, teacher, {m: Wq}, all rows, ones) router_weighted (weight 1), in %."""
        h = self.h
        bf = [t.cuda().bfloat16() for t in Wq]
        num = den = 0.
        n = len(self.cap["x"])
        for f in range(0, n, 64):
            x = self.cap["x"][f:f + 64].cuda()
            target = h._teacher(x, self.tb).double()
            den += float(target.square().sum())
            num += float((h._teacher(x, bf).double() - target).square().sum())
        return round(100 * (num / den) ** .5, 3)

    @torch.no_grad()
    def score(self, Wq, Ws, HG):
        h = self.h
        bf = [t.cuda().bfloat16() for t in Wq]
        num = 0.; den = 0.; es = []
        for f in range(0, len(self.rows), 64):
            ids = self.rows[f:f + 64]
            x = self.cap["x"][ids].cuda()
            p2 = self.p[f:f + 64].cuda().double().square()
            target = h._teacher(x, self.tb).double()
            energy = target.square().sum(-1)
            den += float((energy * p2).sum())
            e = (h._teacher(x, bf).double() - target).square().sum(-1)
            num += float((e * p2).sum())
            es.append((e * p2).cpu())
        es = torch.cat(es)
        m = torch.zeros(len(es), dtype=torch.bool); m[self.top] = True
        out = dict(routed=round(100 * (num / den) ** .5, 3),
                   top10=round(100 * float(es[m].sum() / self.w[m].sum()) ** .5, 3),
                   rest=round(100 * float(es[~m].sum() / self.w[~m].sum()) ** .5, 3),
                   top10_err_share=float(es[m].sum() / es.sum()), top10_energy_share=float(self.w[m].sum() / self.w.sum()))
        proj = {}
        for pi, pn in enumerate(PROJ):
            W = Ws[pi].cuda().float(); E = Wq[pi].cuda().float() - W
            d = {}
            for nm, Hm in (("eH", HG["H"][pi]), ("eV", self.Hval[pi])):
                Hm = Hm.cuda().float()
                d[nm] = float(((E @ Hm) * E).sum() / ((W @ Hm) * W).sum())
            proj[pn] = d
        out["proj"] = proj
        del bf
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True); ap.add_argument("--gpu-gb", type=float, default=12)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--forced", action="store_true", help="also all/forced (every val row, weight 1; slow)")
    ap.add_argument("pairs", nargs="+")
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(a.gpu_gb / 80)
    torch.backends.cuda.matmul.allow_tf32 = False
    import nq19_load, harness as h, nq_decode as D, nq_layer as NL, nq_encode as NE
    sig = dict(NE.PROD["sigma"])
    cap = nq19_load.Capture(root=f"{R}/_stats")
    hcap = NL.open_stats(f"{R}/_stats", f"{R}/_stats_mm", 0.25)
    arms = [x for x in a.arms.split(",") if x]
    for pr in a.pairs:
        L, E = map(int, pr.split(":"))
        od = f"{OUT}/L{L}_E{E}"; os.makedirs(od, exist_ok=True)
        todo = [x for x in arms if a.force or not os.path.exists(f"{od}/{x}.json")]
        if not todo:
            continue
        t0 = time.time()
        dm = cap.expert_data(L, E, "val", source=SRC)
        HG = NL.expert_HG(hcap, L, E)[0]
        Ws = [w.cpu() for w in dm.teacher]
        sc = Scorer(h, dm)
        print(f"L{L} E{E} rows {len(sc.rows)} setup {time.time()-t0:.0f}s", flush=True)
        for arm in todo:
            t1 = time.time()
            head, toks = parse(arm)
            htk = [t for t in toks if t.startswith("alpha") or t.startswith("ctx")]
            toks = [t for t in toks if t not in htk]
            HGa = HG
            if htk:                                   # calibration-H recipe override (T19 glm_H alpha / ctx_mass)
                kwh = dict(alpha=0.25, ctx_mass=0.25)
                for t in htk:
                    kwh["alpha" if t.startswith("alpha") else "ctx_mass"] = float(t[5:] if t.startswith("alpha") else t[3:])
                HGa = hcap.glm_H(L, E, **kwh)
            rtk = [t for t in toks if t.startswith("drot") or t.startswith("xrot")]
            toks = [t for t in toks if t not in rtk]
            Wa, HGa, rots = apply_rots(Ws, HGa, rtk)
            if head.startswith("X"):
                dense, meta = exl3_arm(h, Wa, HGa, head, toks, sig)
            else:
                dense, meta = nq_arm(NE, D, Wa, HGa, head, toks, L, E)
            for Lv, q in dense.items():                          # W = W' R  (x' = R x)
                for pi_, Rm in rots.items():
                    for j in ((0, 1) if pi_ == 0 else (2,)):
                        q[j] = q[j].float() @ Rm
            if rots:
                meta["rot"] = rtk
            ev = {f"L{Lv}": sc.score(q, Ws, HG) for Lv, q in dense.items()}
            if a.forced:
                for Lv, q in dense.items():
                    ev[f"L{Lv}"]["forced"] = sc.forced(q)
            res = dict(layer=L, expert=E, arm=arm, eval=ev, meta=meta, s=round(time.time() - t1),
                       time=time.strftime("%Y-%m-%d %H:%M:%S"))
            json.dump(res, open(f"{od}/{arm}.json", "w"), indent=1)
            print(f"L{L} E{E} {arm:24s} " + " ".join(f"{k} {v['routed']:.3f} (top10 {v['top10']:.2f} rest {v['rest']:.2f})"
                                                   for k, v in ev.items()) + f" {time.time()-t1:.0f}s", flush=True)
            del dense; torch.cuda.empty_cache()
        del dm, HG, sc; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
