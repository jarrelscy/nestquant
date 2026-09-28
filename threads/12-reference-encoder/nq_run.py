"""Thread 12 driver (v1, kernel-exact ref15 fold): NestQuant + same-H EXL3 anchors, scored with harness.evaluate().

python nq_run.py --experts 16:36,49:92 [--variants anchors,rates,k3cmp,sg4,split] [--mimo]
One reference fit per expert (base = per-ring sign + two-sided G + blend 0.3, residual uniform K2) produces the base;
every rate variant re-fits ONLY the level-4 residual on that frozen base (base plane bytes asserted identical).
Results: $NQ_RES (default results_v1)/L{L}_E{E}.json. Artifacts: /tmp/nestquant/12-reference-encoder/v1.
"""
import os, sys, json, time, argparse
os.environ.setdefault("OMP_NUM_THREADS", "16")
import torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nq_decode as D
import nq_encode as NE
import harness as h

SCR = "/tmp/nestquant/12-reference-encoder"
RES = os.environ.get("NQ_RES", f"{HERE}/results_v1")
ART = f"{SCR}/v1"
PROJ = ["gate", "up", "down"]
SIG = {"gate": 0.5, "up": 0.5, "down": 1.0}          # thread-08 selected damping
AXIS = {"gate": "n", "up": "n", "down": "k"}          # TP shard axis in (k=in, n=out) layout


def nt(G):
    return G / G.diagonal().mean()


@torch.no_grad()
def glm_H(data, L, E):
    """Thread-08 recipe: H = 0.25 nt(sum (p x)(p x)^T) + 0.75 nt(sum x x^T) from the training sample (count 1);
    down from bf16 hidden silu(g x)*(u x). G (thread 06) = diag(same mix of sum w c^2)^0.5 for gate/up."""
    path = f"{SCR}/H_l{L}_e{E}.pt"
    if os.path.exists(path):
        return torch.load(path)
    ts = torch.load(f"{h.GLM_RUN.format(L=L)}/statistics/l{L}_e{E}_training_sample.pt", weights_only=True, mmap=True)
    g, u, d = data.teacher
    acc = {k: torch.zeros(6144 if k == "x" else 2048, 6144 if k == "x" else 2048, device="cuda", dtype=torch.float64)
           for k in []}
    S = {}
    for key, dim in (("x", 6144), ("a", 2048)):
        for w in ("p", "u"):
            S[key + w] = torch.zeros(dim, dim, device="cuda", dtype=torch.float64)
    Od = {k: torch.zeros(2048, device="cuda", dtype=torch.float64) for k in ("gp", "gu", "up", "uu")}
    x_all, p_all = ts["x"], ts["p"]
    for i in range(0, len(x_all), 2048):
        x = x_all[i:i + 2048].cuda(); p = p_all[i:i + 2048].cuda().float()[:, None]
        hid = (F.silu(F.linear(x.bfloat16(), g.bfloat16())) * F.linear(x.bfloat16(), u.bfloat16())).float()
        xf = x.float()
        for w, pp in (("p", p), ("u", torch.ones_like(p))):
            a = xf * pp; S["x" + w] += (a.T @ a).double()
            b = hid * pp; S["a" + w] += (b.T @ b).double()
        gx, ux = F.linear(xf, g), F.linear(xf, u); sig = gx.sigmoid()
        cg = ux * sig * (1 + gx * (1 - sig)); cu = F.silu(gx)
        Od["gp"] += (cg * p).square().sum(0).double(); Od["gu"] += cg.square().sum(0).double()
        Od["up"] += (cu * p).square().sum(0).double(); Od["uu"] += cu.square().sum(0).double()
    mix = lambda A, B: (0.25 * nt(A) + 0.75 * nt(B)).float()
    Hx = mix(S["xp"], S["xu"]); Ha = mix(S["ap"], S["au"])
    dmix = lambda A, B: (0.25 * A / A.mean() + 0.75 * B / B.mean()).float()
    Gg = torch.diag(dmix(Od["gp"], Od["gu"]).clamp_min(1e-30).pow(0.5))
    Gu = torch.diag(dmix(Od["up"], Od["uu"]).clamp_min(1e-30).pow(0.5))
    out = dict(H=[Hx.cpu(), Hx.cpu(), Ha.cpu()], G=[Gg.cpu(), Gu.cpu(), None])
    torch.save(out, path)
    return out


class Book:
    def __init__(self, path):
        self.path = path
        self.R = json.load(open(path)) if os.path.exists(path) else {}

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + ".tmp"; json.dump(self.R, open(tmp, "w"), indent=1); os.replace(tmp, self.path)


def evaluate_into(book, data, methods, extra=None):
    """methods {name: [g,u,d] cpu fp32}; evaluates in groups of 4, writes table rows into book."""
    names = list(methods)
    for i in range(0, len(names), 4):
        grp = {n: [w.cuda() for w in methods[n]] for n in names[i:i + 4]}
        tb = h.table(h.evaluate(data, grp, groups=True))
        for n in grp:
            book.R.setdefault("eval", {})[n] = tb[n]
            if extra and n in extra:
                book.R.setdefault("info", {})[n] = extra[n]
        del grp; torch.cuda.empty_cache()
        book.save()


INNER = 2
BASE_VAR = os.environ.get("NQ_BASE_VAR", "sign")
TARGETS = tuple(float(x) for x in os.environ.get("NQ_TARGETS", "4.0625,4.09375,4.125,4.25").split(","))
UNITS = 768                                            # units per TP shard (all three projections)


def bits_expert(info, L):
    return sum(info[p]["bits"][L] for p in PROJ) / 3          # equal-size projections


def frac_for(ref_bits, target, K_hi, K=2):
    """Positional/mask fraction of units (whole units per shard) so the L4 rate hits `target` bpw."""
    f = (target - ref_bits) / (K_hi - K)
    return max(0, round(f * UNITS)) / UNITS


def base_identical(a, b):
    ok = all(torch.equal(x, y) for x, y in zip(a["base"]["shards"], b["base"]["shards"]))
    if a["base"].get("var") is not None:
        ok &= all(torch.equal(x, y) for x, y in zip(a["base"]["var"], b["base"]["var"]))
    return bool(ok and torch.equal(a["base"]["suh"], b["base"]["suh"]) and torch.equal(a["base"]["svh"], b["base"]["svh"]))


class Fit:
    """Reference fit (base) of one expert + residual-only rate variants on the frozen base."""
    def __init__(self, Ps, base_var, lam=0.3, inner=INNER, tag=""):
        self.Ps, self.bv, self.lam, self.inner, self.tag = Ps, base_var, lam, inner, tag
        self.ref = {}

    def run(self, name, rules=None, masks=None):
        """rules {proj: res_rule} (None = reference uniform K2 base fit). -> (dense {2,4}, info, planes)."""
        dense = {2: [], 4: []}; info = {}; art = {}
        for pn in PROJ:
            P = self.Ps[pn]
            kw = dict(shard_axis=AXIS[pn], lam=self.lam, base_var=self.bv, inner=self.inner)
            if rules is not None:
                r = self.ref[pn]
                kw.update(base=r["base"], base_scales=r["scales"], res_rule=rules[pn],
                          mask_flat=None if masks is None else masks[pn])
            planes, dn, inf, enc, sc = NE.encode_projection(P, **kw)
            rot = D.rotated_levels(planes)
            inf["bitexact"] = {L: bool(torch.equal(D.decode_matrix(planes, L, rot=rot), dn[L])) for L in (2, 4)}
            inf["ref15_mismatch"] = D.xcheck_ref15(planes, 8)
            inf["plane_bytes"] = D.plane_bytes(planes)
            if rules is None:
                self.ref[pn] = dict(base=NE.frozen_base(enc), scales=sc[2], planes=planes, cost4=enc["cost4"],
                                    dense2=dn[2].cpu())
            else:
                inf["base_identical"] = base_identical(self.ref[pn]["planes"], planes)
                inf["L2_equal"] = bool(torch.equal(dn[2].cpu(), self.ref[pn]["dense2"]))
            for L in (2, 4):
                dense[L].append(dn[L].cpu())
            info[pn] = inf; art[pn] = planes
            del enc, rot; torch.cuda.empty_cache()
        os.makedirs(ART, exist_ok=True)
        torch.save(art, f"{ART}/{self.tag}{name}.pt")
        return dense, info


def summary_info(info, L):
    keep = ("bits", "proxy_rot", "time", "bitexact", "ref15_mismatch", "stream_mismatch", "base_identical", "L2_equal",
            "delta_mean", "Mb_mean", "Mb_min", "N0_frac", "K_frac", "gsr")
    return dict(bpw=bits_expert(info, L), artifact_bpw=bits_expert(info, "artifact"),
                proj={p: {k: info[p][k] for k in keep if k in info[p]} for p in PROJ})


def copy_anchors(book, L, E):
    for d in ("results", "results_b", "results_sg4"):
        f = f"{HERE}/{d}/L{L}_E{E}.json"
        if os.path.exists(f):
            R = json.load(open(f))
            for n in ("EXL3-2", "EXL3-4", "NVFP4"):
                if n in R.get("eval", {}) and n not in book.R.get("eval", {}):
                    book.R.setdefault("eval", {})[n] = R["eval"][n]
                    book.R.setdefault("info", {})[n] = dict(R.get("info", {}).get(n, {}), copied_from=f)


def exl3_matched(Ws, HG, target):
    """EXL3 with K5 on the lowest-index (= last-processed) 16-blocks so the bpw (incl. fp16 scales) ~= target."""
    q, bpws = [], []
    for pi, pn in enumerate(PROJ):
        n_out, k_in = Ws[pi].shape
        nb = k_in // 16
        sc = 16 * (k_in + n_out) / (k_in * n_out)
        m = max(0, round((target - 4 - sc) * nb))
        Ks = [5 if j < m else 4 for j in range(nb)]
        Wq, inf = h.quantize_exl3_like(Ws[pi], HG["H"][pi], Ks, count=1, sigma_reg=SIG[pn])
        q.append(Wq.cpu()); bpws.append(inf["bpw"]); h.free_scratch()
    return q, sum(bpws) / 3


def run_glm(L, E, variants):
    os.makedirs(SCR, exist_ok=True)
    book = Book(f"{RES}/L{L}_E{E}.json")
    t0 = time.time()
    data = h.load_expert(L, E)
    HG = glm_H(data, L, E)
    Ws = data.teacher
    ev = book.R.setdefault("eval", {})
    log = lambda m: print(f"[{L}:{E}] {m} {time.time()-t0:.0f}s", flush=True)
    methods, extra = {}, {}

    def flush():
        nonlocal methods, extra
        if methods:
            evaluate_into(book, data, methods, extra)
        methods, extra = {}, {}
    # ---- anchors
    copy_anchors(book, L, E)
    if "anchors" in variants:
        for K in (2, 4):
            nm = f"EXL3-{K}"
            if nm not in ev:
                q = []
                for pi, pn in enumerate(PROJ):
                    Wq, _ = h.quantize_exl3_like(Ws[pi], HG["H"][pi], K, count=1, sigma_reg=SIG[pn])
                    q.append(Wq.cpu()); h.free_scratch()
                methods[nm] = q; extra[nm] = dict(bpw=K + 16 * (6144 + 2048) / (6144 * 2048))
        if "NVFP4" not in ev:
            methods["NVFP4"] = [w.cpu() for w in h.load_nvfp4(f"{h.GLM_RUN.format(L=L)}/nvfp4_e{E}/weights.pt", data)]
            extra["NVFP4"] = dict(bpw=4.5)
        flush(); log("anchors")
    Ps = {pn: NE.prep(Ws[pi], HG["H"][pi], 1, SIG[pn], G=HG["G"][pi]) for pi, pn in enumerate(PROJ)}
    log("prep")
    stacks = [("nq", BASE_VAR)] + ([("nq_sg4", "sg4")] if "sg4" in variants else [])
    for sname, bv in stacks:
        fit = Fit(Ps, bv, tag=f"L{L}_E{E}_")
        dense, info = fit.run(sname)
        ref_bits = bits_expert(info, 4)
        for Lv in (2, 4):
            nm = f"{sname}/L{Lv}"
            if nm in ev:                               # re-run for extra rates: the ref fit is deterministic
                continue
            methods[nm] = dense[Lv]; extra[nm] = summary_info(info, Lv)
        book.R.setdefault("planes", {})[sname] = {p: info[p]["plane_bytes"] for p in PROJ}
        flush(); log(f"{sname} ref fit L4 {ref_bits:.4f} bitexact {[info[p]['bitexact'] for p in PROJ]} "
                     f"ref15 {[info[p]['ref15_mismatch'] for p in PROJ]}")
        # ---- rate variants on the frozen base (residual only)
        rv = []
        if "rates" in variants:
            if "no40" not in variants:
                rv.append((f"{sname}_r4.0", dict(kind="pos", K=2, K_hi=1.5, frac=frac_for(ref_bits, 4.0, 1.5)), None))
            for T in TARGETS:
                rv.append((f"{sname}_r{T}", dict(kind="pos", K=2, K_hi=2.5, frac=frac_for(ref_bits, T, 2.5)), None))
        if "k3cmp" in variants and sname == "nq":
            for T in (4.125, 4.25):
                f3 = frac_for(ref_bits, T, 3)
                rv.append((f"{sname}_pos3_r{T}", dict(kind="pos", K=2, K_hi=3, frac=f3), None))
                rv.append((f"{sname}_mask3_r{T}", dict(kind="mask", K=2, K_hi=3, frac=f3),
                           {p: NE.select_mask(fit.ref[p]["cost4"], AXIS[p], f3) for p in PROJ}))
        if "split" in variants and sname == "nq":     # low priority (thread 16: worse at 4.0): g/u 1.75 ~ down 2.5
            pass
        for nm, rule, masks in rv:
            if f"{nm}/L4" in ev:
                continue
            dense, info = fit.run(nm, rules={p: rule for p in PROJ}, masks=masks)
            methods[f"{nm}/L4"] = dense[4]
            extra[f"{nm}/L4"] = dict(summary_info(info, 4), rule=rule, L2_bpw=bits_expert(info, 2),
                                     L2_equals_base=all(info[p]["L2_equal"] and info[p]["base_identical"] for p in PROJ))
            book.R.setdefault("planes", {})[nm] = {p: info[p]["plane_bytes"] for p in PROJ}
            flush(); log(f"{nm} L4 {bits_expert(info, 4):.4f} base+L2 identical "
                         f"{[info[p]['base_identical'] and info[p]['L2_equal'] for p in PROJ]} "
                         f"bitexact {[info[p]['bitexact'] for p in PROJ]} ref15 {[info[p]['ref15_mismatch'] for p in PROJ]}")
            h.free_scratch()
        del fit; torch.cuda.empty_cache()
    # ---- matched-rate EXL3 anchors
    if "anchors" in variants:
        for T in (4.0221,) + TARGETS:
            nm = f"EXL3-4+{T}"
            if nm in ev:
                continue
            q, bpw = exl3_matched(Ws, HG, T)
            methods[nm] = q; extra[nm] = dict(bpw=bpw, rule="K5 on lowest-index 16-blocks")
            flush(); log(f"{nm} bpw {bpw:.4f}")
    book.R["time_s"] = time.time() - t0
    book.save()
    for p in Ps:
        NE.free(Ps[p])
    h.free_scratch()
    return book


MIMO_SRC = "/tmp/mimo-a100/data/jarrel/mimo-exl3-sequential/source"          # read only
MIMO_STATS = h.ORBIT + "/runs/full55_statistics/l{L}_e{E}.pt"
MIMO_CAPS = {"id": h.ORBIT + "/runs/native_id_control_v1_capture/layer_{L}.pt",
             "ood": h.ORBIT + "/runs/ood_controlled_v1_capture/layer_{L}.pt"}


def run_mimo(L=55, E=70):
    """MiMo L55 E70 (no down split): H = router-p^2 grams / training_rows, sigma 0.03, G two-sided, lambda 0 (seq)."""
    book = Book(f"{RES}/mimo_L{L}_E{E}.json")
    t0 = time.time()
    data = h.load_expert(L, E, source=MIMO_SRC, statistics=MIMO_STATS.format(L=L, E=E), capture=MIMO_CAPS["id"].format(L=L))
    caps = {"id": data.capture, "ood": torch.load(MIMO_CAPS["ood"].format(L=L), weights_only=True, mmap=True)}
    st = data.stats; cnt = st["metadata"]["training_rows"]
    Hx, Ha = st["grams"][0].float(), st["grams"][1].float()
    dg = [st["outputs"][i].float().diagonal().clamp_min(1e-30).pow(0.5) for i in range(2)]
    Ws = [w.float() for w in data.teacher]
    sig = 0.03

    def ev(methods, extra):
        for dom, cap in caps.items():
            data.capture = cap
            names = list(methods)
            for i in range(0, len(names), 4):
                grp = {n: [w.cuda() for w in methods[n]] for n in names[i:i + 4]}
                tb = h.table(h.evaluate(data, grp, groups=False), domains=("all",))
                for n in grp:
                    book.R.setdefault("eval", {}).setdefault(n, {}).update({f"{dom}/{kk.split('/')[1]}": v for kk, v in tb[n].items()})
                    if n in extra:
                        book.R.setdefault("info", {})[n] = extra[n]
                del grp; torch.cuda.empty_cache()
        data.capture = caps["id"]
        book.save()
    old = f"{HERE}/results/mimo_L{L}_E{E}.json"
    if os.path.exists(old):
        R = json.load(open(old))
        for n in ("EXL3-2", "EXL3-4"):
            if n in R["eval"]:
                book.R.setdefault("eval", {})[n] = R["eval"][n]
    methods, extra = {}, {}
    for K in (2, 4):
        if f"EXL3-{K}" not in book.R.get("eval", {}):
            q = []
            for pi in range(3):
                Wq, _ = h.quantize_exl3_like(Ws[pi], [Hx, Hx, Ha][pi], K, count=cnt, sigma_reg=sig)
                q.append(Wq.cpu()); h.free_scratch()
            methods[f"EXL3-{K}"] = q; extra[f"EXL3-{K}"] = dict(bpw=K + 16 * (6144 + 2048) / (6144 * 2048))
    if methods:
        ev(methods, extra)
    Ps = {}
    for pi, pn in enumerate(PROJ):
        G = torch.diag(dg[pi]) if pi < 2 else None
        Ps[pn] = NE.prep(Ws[pi], [Hx, Hx, Ha][pi], cnt, sig, G=G, sigma_out=sig)
    fit = Fit(Ps, BASE_VAR, lam=0.0, tag=f"mimo_L{L}_E{E}_")
    dense, info = fit.run("nq")
    methods = {f"nq/L{Lv}": dense[Lv] for Lv in (2, 4)}
    extra = {f"nq/L{Lv}": summary_info(info, Lv) for Lv in (2, 4)}
    book.R.setdefault("planes", {})["nq"] = {p: info[p]["plane_bytes"] for p in PROJ}
    ev(methods, extra)
    print(f"[mimo] nq {time.time()-t0:.0f}s bitexact {[info[p]['bitexact'] for p in PROJ]}", flush=True)
    book.R["time_s"] = time.time() - t0
    book.save()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", default="16:36")
    ap.add_argument("--variants", default="anchors,rates,k3cmp")
    ap.add_argument("--mimo", action="store_true")
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(12 / 80)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(16)
    if a.mimo:
        run_mimo(); return
    for spec in a.experts.split(","):
        L, E = map(int, spec.split(":"))
        run_glm(L, E, a.variants.split(","))


if __name__ == "__main__":
    main()
