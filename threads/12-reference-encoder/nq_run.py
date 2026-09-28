"""Thread 12 driver: fit NestQuant v0 + same-H EXL3 anchors on GLM experts, score with harness.evaluate().

python nq_run.py --experts 16:36 [--variants pilot,k3,closers] [--smoke]
Results: results/L{L}_E{E}.json (checkpointed after every method). Artifacts: /tmp/nestquant/12-reference-encoder.
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
RES = os.environ.get("NQ_RES", f"{HERE}/results")
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


def nq_variant(Ps, HG, name, lam=0.3, k3=None, cands=(("mul1",),), delta_per="unit", gain=1.0,
               keep_artifact=False, check=False, tag="", inner=None, base_var=None):
    """Encode all three projections with shared prep Ps -> (dense {2,3,4: [g,u,d] cpu}, info, encs)."""
    dense = {L: [] for L in (2, 4)}
    info = {}; encs = {}; art = {}
    for pi, pn in enumerate(PROJ):
        planes, dn, inf, P, enc = NE.encode_projection(None, None, None, None, P=Ps[pn], lam=lam, shard_axis=AXIS[pn],
                                                     k3=None if k3 is None else k3[pn], cands=cands,
                                                     delta_per=delta_per, gain=gain,
                                                     inner=INNER if inner is None else inner, base_var=base_var)
        if check:
            rot = D.rotated_levels(planes)
            for L in (2, 4):
                Wd = D.decode_matrix(planes, L, rot=rot)
                inf.setdefault("bitexact", {})[L] = bool(torch.equal(Wd, dn[L]))
            del rot
        for L in (2, 4):
            dense[L].append(dn[L].cpu())
        info[pn] = {k: v for k, v in inf.items()}
        info[pn]["plane_bytes"] = D.plane_bytes(planes)
        encs[pn] = dict(cost4=enc["cost4"].cpu(), cost2=enc["cost2"].cpu())
        art[pn] = planes
        del enc, dn
        torch.cuda.empty_cache()
    if keep_artifact:
        torch.save(art, f"{SCR}/{tag}{name}.pt")
    return dense, info, encs


def bits_expert(info, L):
    nw = {"gate": 2048 * 6144, "up": 2048 * 6144, "down": 2048 * 6144}
    return sum(info[p]["bits"][L] * nw[p] for p in PROJ) / sum(nw.values())


def run_glm(L, E, variants, smoke=False):
    os.makedirs(SCR, exist_ok=True)
    book = Book(f"{RES}/L{L}_E{E}.json")
    t0 = time.time()
    data = h.load_expert(L, E)
    HG = glm_H(data, L, E)
    Ws = data.teacher
    print(f"[{L}:{E}] data+H {time.time()-t0:.0f}s", flush=True)
    methods, extra = {}, {}
    # ---- anchors
    if "anchors" in variants:
        for K in (2, 4):
            nm = f"EXL3-{K}"
            if nm in book.R.get("eval", {}):
                continue
            q = []
            for pi, pn in enumerate(PROJ):
                Wq, inf = h.quantize_exl3_like(Ws[pi], HG["H"][pi], K, count=1, sigma_reg=SIG[pn])
                q.append(Wq.cpu())
            h.free_scratch()
            methods[nm] = q; extra[nm] = dict(bpw=K + 16 * (6144 + 2048) / (6144 * 2048))
        if "NVFP4" not in book.R.get("eval", {}):
            methods["NVFP4"] = [w.cpu() for w in h.load_nvfp4(f"{h.GLM_RUN.format(L=L)}/nvfp4_e{E}/weights.pt", data)]
            extra["NVFP4"] = dict(bpw=4.5)
        evaluate_into(book, data, methods, extra); methods, extra = {}, {}
        print(f"[{L}:{E}] anchors {time.time()-t0:.0f}s", flush=True)
    # ---- shared preprocessing
    Ps = {}
    for pi, pn in enumerate(PROJ):
        G = HG["G"][pi]
        Ps[pn] = NE.prep(Ws[pi], HG["H"][pi], 1, SIG[pn], G=G)
    print(f"[{L}:{E}] prep {time.time()-t0:.0f}s", flush=True)

    def add(name, dense, info, levels=(2, 4)):
        for Lv in levels:
            nm = f"{name}/L{Lv}"
            methods[nm] = dense[Lv]
            extra[nm] = dict(bpw=bits_expert(info, Lv), artifact_bpw=bits_expert(info, "artifact"),
                             **({"proj": {p: {k: info[p][k] for k in ("bits", "proxy_rot", "time", "gs", "gsr", "bitexact", "cost4_mean")
                                             if k in info[p]} for p in PROJ}} if Lv == 4 else {}))
        if Lv == 4 or True:
            book.R.setdefault("planes", {})[name] = {p: info[p]["plane_bytes"] for p in PROJ}

    # ---- pilot (GLM default: blend 0.3, mul1 residual, delta per 16x128)
    pilot_costs = None
    t = time.time()
    dense, info, encs = nq_variant(Ps, HG, "nq", check=True, keep_artifact=True, tag=f"L{L}_E{E}_")
    pilot_costs = {p: encs[p]["cost4"] for p in PROJ}
    print(f"[{L}:{E}] pilot encode {time.time()-t:.0f}s bitexact "
          f"{[info[p].get('bitexact') for p in PROJ]} proxy {[round(info[p]['proxy_rot'][4],6) for p in PROJ]}", flush=True)
    add("nq", dense, info)
    evaluate_into(book, data, methods, extra); methods, extra = {}, {}
    if smoke:
        return book
    # ---- 3-bit residual on the top-cost units (per shard)
    if "k3" in variants:
        for f in (0.0625, 0.125, 0.25):
            k3 = {p: NE.select_k3(pilot_costs[p].cuda(), AXIS[p], f) for p in PROJ}
            dense, info, _ = nq_variant(Ps, HG, f"nq_k3_{f}", k3=k3, check=True)
            add(f"nq_k3_{f}", dense, info, levels=(4,))
            evaluate_into(book, data, methods, extra); methods, extra = {}, {}
            h.free_scratch()
            print(f"[{L}:{E}] k3 {f} done {time.time()-t0:.0f}s", flush=True)
    # ---- closers at 4.0 bpw
    if "closers" in variants:
        for name, kw in (("nq_sign", dict(cands=(("mul1",), ("mul1", -1.0)))),
                         ("nq_ring_delta", dict(delta_per="ring")),
                         ("nq_cb6", dict(cands=tuple((c, s) for c in ("mul1", "mcg", "3inst") for s in (1.0, -1.0))))):
            dense, info, _ = nq_variant(Ps, HG, name, check=True, **kw)
            add(name, dense, info, levels=(4,))
            evaluate_into(book, data, methods, extra); methods, extra = {}, {}
            h.free_scratch()
            print(f"[{L}:{E}] {name} done {time.time()-t0:.0f}s", flush=True)
    if "combo" in variants:
        for f in (0.0625, 0.125, 0.25):
            k3 = {p: NE.select_k3(pilot_costs[p].cuda(), AXIS[p], f) for p in PROJ}
            nm = f"nq_sign_ring_k3_{f}"
            dense, info, _ = nq_variant(Ps, HG, nm, k3=k3, cands=(("mul1",), ("mul1", -1.0)), delta_per="ring")
            add(nm, dense, info, levels=(4,))
            evaluate_into(book, data, methods, extra); methods, extra = {}, {}
            h.free_scratch()
            print(f"[{L}:{E}] {nm} done {time.time()-t0:.0f}s", flush=True)
    if "sg4" in variants:        # thread 17 winner: per-ring sign+gain base variants (3 bits / 256-weight ring)
        for nm, kw in (("nq_sg4", dict(base_var="sg4")), ("nq_sign4", dict(base_var="sign"))):
            dense, info, _ = nq_variant(Ps, HG, nm, check=True, **kw)
            add(nm, dense, info)
            evaluate_into(book, data, methods, extra); methods, extra = {}, {}
            h.free_scratch()
            print(f"[{L}:{E}] {nm} done {time.time()-t0:.0f}s bitexact {[info[p].get('bitexact') for p in PROJ]}", flush=True)
    if "k3fine" in variants:
        for f in (0.1875,):
            k3 = {p: NE.select_k3(pilot_costs[p].cuda(), AXIS[p], f) for p in PROJ}
            nm = f"nq_k3_{f}"
            dense, info, _ = nq_variant(Ps, HG, nm, k3=k3, check=True)
            add(nm, dense, info, levels=(4,))
            evaluate_into(book, data, methods, extra); methods, extra = {}, {}
            h.free_scratch()
            print(f"[{L}:{E}] {nm} done {time.time()-t0:.0f}s", flush=True)
    if "stack" in variants:      # thread-17 sg4 base + residual sign choice + K3 residual
        for f in (0.0, 0.125, 0.15625):
            k3 = {p: NE.select_k3(pilot_costs[p].cuda(), AXIS[p], f) for p in PROJ} if f else None
            nm = f"nq_sg4_sign_k3_{f}"
            dense, info, _ = nq_variant(Ps, HG, nm, k3=k3, cands=(("mul1",), ("mul1", -1.0)), base_var="sg4", check=True)
            add(nm, dense, info, levels=(2, 4) if f == 0.0 else (4,))
            evaluate_into(book, data, methods, extra); methods, extra = {}, {}
            h.free_scratch()
            print(f"[{L}:{E}] {nm} done {time.time()-t0:.0f}s bitexact {[info[p].get('bitexact') for p in PROJ]}", flush=True)
    if "inner0" in variants:
        dense, info, _ = nq_variant(Ps, HG, "nq_inner0", inner=0)
        add("nq_inner0", dense, info)
        evaluate_into(book, data, methods, extra); methods, extra = {}, {}
    if "seq" in variants:
        dense, info, _ = nq_variant(Ps, HG, "nq_seq", lam=0.0)
        add("nq_seq", dense, info)
        evaluate_into(book, data, methods, extra); methods, extra = {}, {}
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


def run_mimo(L=55, E=70, keep=True):
    """MiMo L55 E70, level 2 (and 4) vs same-H EXL3, with/without the thread-01 down split (act-order-shard, 1/3 per shard).
    H = router-p^2 grams / training_rows, sigma 0.03; G = diag(outputs)^0.5 (two-sided gate/up); lambda = 0 (seq)."""
    os.makedirs(SCR, exist_ok=True)
    book = Book(f"{RES}/mimo_L{L}_E{E}.json")
    t0 = time.time()
    data = h.load_expert(L, E, source=MIMO_SRC, statistics=MIMO_STATS.format(L=L, E=E), capture=MIMO_CAPS["id"].format(L=L))
    caps = {"id": data.capture, "ood": torch.load(MIMO_CAPS["ood"].format(L=L), weights_only=True, mmap=True)}
    st = data.stats; cnt = st["metadata"]["training_rows"]
    Hx, Ha = st["grams"][0].float(), st["grams"][1].float()
    dg = [st["outputs"][i].float().diagonal().clamp_min(1e-30).pow(0.5) for i in range(2)]
    Ws = [w.float() for w in data.teacher]
    sig = 0.03
    srt = torch.argsort(Ha.diagonal()); perm = torch.cat([srt[j::8] for j in range(8)])
    k = Ha.shape[0]; nblk = k // 16; shard = k // 8

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

    for split in (False, True):
        tag = "_split" if split else ""
        if split:
            pm = perm
            Wg, Wu_, Wd = Ws[0][pm], Ws[1][pm], Ws[2][:, pm]
            Hd = Ha[pm][:, pm]; Gg, Gu = dg[0][pm], dg[1][pm]
        else:
            pm = None; Wg, Wu_, Wd = Ws; Hd = Ha; Gg, Gu = dg
        Wl, Hl = [Wg, Wu_, Wd], [Hx, Hx, Hd]
        unperm = (lambda q: [q[0][torch.argsort(pm)], q[1][torch.argsort(pm)], q[2][:, torch.argsort(pm)]]) if split else (lambda q: q)
        # ---- anchors: EXL3-2/4 same H (split: K list per 16-block, low half of each shard K-1, high half K+1)
        methods, extra = {}, {}
        for K in (2, 4):
            nm = f"EXL3-{K}{tag}"
            if nm in book.R.get("eval", {}):
                continue
            q = []
            for pi in range(3):
                Kp = [K - 1 if (j % (shard // 16)) < shard // 32 else K + 1 for j in range(nblk)] if (split and pi == 2) else K
                Wq, _ = h.quantize_exl3_like(Wl[pi], Hl[pi], Kp, count=cnt, sigma_reg=sig)
                q.append(Wq.cpu()); h.free_scratch()
            methods[nm] = unperm(q); extra[nm] = dict(bpw=K + 16 * (6144 + 2048) / (6144 * 2048))
        if methods:
            ev(methods, extra)
        print(f"[mimo{tag}] anchors {time.time()-t0:.0f}s", flush=True)
        # ---- NestQuant (lambda 0, inner 2)
        art = {}; dense = {2: [], 4: []}; info = {}
        for pi, pn in enumerate(PROJ):
            G = torch.diag(Gg if pi == 0 else Gu) if pi < 2 else None
            P = NE.prep(Wl[pi], Hl[pi], cnt, sig, G=G, sigma_out=sig)
            bK = [1, 3] * (k // 256) if (split and pi == 2) else None
            planes, dn, inf, _, enc = NE.encode_projection(None, None, None, None, P=P, lam=0.0, shard_axis=AXIS[pn],
                                                         base_K=bK, inner=INNER)
            rot = D.rotated_levels(planes)
            inf["bitexact"] = {Lv: bool(torch.equal(D.decode_matrix(planes, Lv, rot=rot), dn[Lv])) for Lv in (2, 4)}
            for Lv in (2, 4):
                dense[Lv].append(dn[Lv].cpu())
            inf["plane_bytes"] = D.plane_bytes(planes)
            info[pn] = inf; art[pn] = planes
            NE.free(P); del enc, rot; torch.cuda.empty_cache()
        if split:
            art["meta"] = dict(inter_perm=perm.tolist())
        if keep:
            torch.save(art, f"{SCR}/mimo_L{L}_E{E}_nq{tag}.pt")
            # decode-from-artifact check at expert level (includes the inter_perm un-permute)
            for Lv in (2, 4):
                Wdec = D.decode_expert(art, Lv)
                info.setdefault("expert_bitexact", {})[Lv] = all(torch.equal(a.cpu(), b) for a, b in zip(Wdec, unperm(dense[Lv])))
        print(f"[mimo{tag}] nq encode {time.time()-t0:.0f}s bitexact {[info[p]['bitexact'] for p in PROJ]} "
              f"expert {info.get('expert_bitexact')}", flush=True)
        methods, extra = {}, {}
        for Lv in (2, 4):
            nm = f"nq{tag}/L{Lv}"
            methods[nm] = unperm(dense[Lv])
            extra[nm] = dict(bpw=bits_expert(info, Lv), artifact_bpw=bits_expert(info, "artifact"),
                             proj={p: {kk: info[p][kk] for kk in ("bits", "proxy_rot", "time", "bitexact") if kk in info[p]} for p in PROJ},
                             expert_bitexact=info.get("expert_bitexact"))
        book.R.setdefault("planes", {})[f"nq{tag}"] = {p: info[p]["plane_bytes"] for p in PROJ}
        ev(methods, extra)
        h.free_scratch()
    book.R["time_s"] = time.time() - t0
    book.save()
    return book


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", default="16:36")
    ap.add_argument("--variants", default="anchors,k3,closers")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--mimo", action="store_true")
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(12 / 80)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(16)
    if a.mimo:
        run_mimo(); return
    for spec in a.experts.split(","):
        L, E = map(int, spec.split(":"))
        run_glm(L, E, a.variants.split(","), smoke=a.smoke)


if __name__ == "__main__":
    main()
