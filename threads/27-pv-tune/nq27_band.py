"""T27 band pilot: sequential layer-by-layer PV of the routed experts on PROPAGATED inputs (T18 hdump).

  prep  python nq27_band.py prep --layer L [--dump DIR]
        -> OUT/band/prep/{tag}/L{L}/{D.bf16, meta.json}: D = (d_ref - shared) - R_fp8 per token, fp32 -> bf16, where
           d_ref = h_out_fp8 - h_mid_nqdef (T18), shared = shared expert on x_nqdef and
           R_fp8 = sum_k p_k f^FP8_{E_k}(x_nqdef) (the FP8 routed experts on the propagated input, nqdef routing).
           D is the part of the FP8 layer output that FP8 experts on the drifted input do NOT produce (upstream drift).
  tune  python nq27_band.py tune --layer L --target same|ref --arm NAME [--dump DIR] [--w W --nw NW] [tuner flags]
        per expert E of layer L (E % NW == W): routed rows of x_nqdef (ids_nqdef/p_nqdef), split by window
        (window % 10 == 0 held out), target
          same: f^FP8_E(x_nqdef)                               (same_input)
          ref : f^FP8_E(x_nqdef) + p_E * D / sum_k p_k^2       (min-norm share of the FP8 layer output: the 8 routed
                experts' shares sum to D exactly, so sum_k p_k (target_k) = d_ref - shared = FP8's own layer output)
        tuned: su/sv (per level) + U2/U4/V exactly as nq27_tune (a = 0.5 joint over both levels), robust act loss
        (norm + cap-q), forced regularizer on uniform x_nqdef rows (gamma), held-out stop.
        -> OUT/band/{arm}/L{L}/experts/E{E}.pt (+ .json).  Untouched format; decode re-checked through nq_decode.
"""
import os, sys, json, time, argparse
os.environ.setdefault("OMP_NUM_THREADS", "8")
import torch
from safetensors import safe_open

OUT = "/tmp/nestquant/27-pv-tune"
SRC = "/tmp/nestquant/src/glm53-fp8"
ENC = "/tmp/nestquant/nq-encode-v1"
DUMP = "/tmp/nestquant/18-e2e/hdump"
NR = 8


def load(dump, L, name, ranks=range(NR)):
    ts = []
    for r in ranks:
        with safe_open(f"{dump}/L{L}/{name}.r{r}of{NR}.safetensors", "pt") as f:
            ts.append(f.get_tensor(name))
    return torch.cat(ts)


def windows(dump):
    """global window id of every token (rank-major concat order), from the per-rank manifests."""
    out = []
    for r in range(NR):
        m = json.load(open(f"{dump}/manifest.r{r}of{NR}.json"))
        seq = m["seq"]
        for _, ws in m["windows"]:
            for w in ws:
                out.append(torch.full((seq,), w, dtype=torch.int32))
    return torch.cat(out)


def tag_of(dump):
    return os.path.basename(os.path.normpath(dump))


def cmd_prep(a):
    import nq27_tune as T
    from orbit_duet.source import weights
    torch.cuda.set_per_process_memory_fraction(a.gpu_gb / 80)
    od = f"{OUT}/band/prep/{tag_of(a.dump)}/L{a.layer}"; os.makedirs(od, exist_ok=True)
    t0 = time.time()
    x = load(a.dump, a.layer, "x_nqdef"); ids = load(a.dump, a.layer, "ids_nqdef"); p = load(a.dump, a.layer, "p_nqdef")
    Tn = len(x)
    R = torch.zeros(Tn, x.shape[1], dtype=torch.float32)
    for E in range(256):
        rows, sl = torch.where(ids == E)
        if len(rows) == 0:
            continue
        tb = [w.to("cuda").bfloat16() for w in weights(SRC, a.layer, E)]
        for i in range(0, len(rows), 8192):
            r = rows[i:i + 8192]
            y = T.teacher_bf16(x[r].cuda(), tb) * p[r, sl[i:i + 8192]].cuda()[:, None]
            R.index_add_(0, r, y.cpu())
        del tb
    del x
    acc = dict(t=0., d=0., q=0., e=0.); o = 0
    with open(f"{od}/D.bf16.tmp", "wb") as fo:
        for r in range(NR):
            sh = load(a.dump, a.layer, "shared_nqdef", [r]).float()
            tgt = load(a.dump, a.layer, "d_ref_nqdef", [r]).float() - sh             # FP8 layer output's routed part
            n = len(tgt); Rr = R[o:o + n]; o += n
            Dr = tgt - Rr
            q = load(a.dump, a.layer, "moe_out_nqdef", [r]).float() - sh - Rr        # nq routed - FP8 routed
            acc["t"] += float(tgt.double().square().sum()); acc["d"] += float(Dr.double().square().sum())
            acc["q"] += float(q.double().square().sum()); acc["e"] += float((q - Dr).double().square().sum())
            fo.write(Dr.bfloat16().view(torch.int16).numpy().tobytes())
            del sh, tgt, Dr, q
    os.replace(f"{od}/D.bf16.tmp", f"{od}/D.bf16")
    meta = dict(layer=a.layer, dump=a.dump, tokens=Tn,
                rel_drift=(acc["d"] / acc["t"]) ** .5,     # ||D|| / ||FP8 routed target||  (upstream drift share)
                rel_quant=(acc["q"] / acc["t"]) ** .5,     # ||nq routed - FP8 routed (same input)|| / same
                rel_total=(acc["e"] / acc["t"]) ** .5,     # ||nq routed - FP8 layer-output target|| / same
                s=round(time.time() - t0))
    json.dump(meta, open(f"{od}/meta.json", "w"), indent=1)
    print(meta, flush=True)


def cmd_tune(a):
    import nq_decode as D_
    import nq27_tune as T
    from orbit_duet.source import weights
    torch.cuda.set_per_process_memory_fraction(a.gpu_gb / 80)
    torch.backends.cuda.matmul.allow_tf32 = a.tf32
    L = a.layer
    x = load(a.dump, L, "x_nqdef"); ids = load(a.dump, L, "ids_nqdef"); p = load(a.dump, L, "p_nqdef")
    win = windows(a.dump); assert len(win) == len(x)
    Dm = None
    if a.target == "ref":
        pd = f"{OUT}/band/prep/{tag_of(a.dump)}/L{L}/D.bf16"
        Dm = torch.from_file(pd, shared=False, size=x.numel(), dtype=torch.int16).view(torch.bfloat16).view(x.shape)
        sp2 = p.double().square().sum(1)                                              # sum_k p_k^2 per token
    # forced rows: uniform tokens of the propagated input (same window split), target = FP8 expert (same_input)
    g = torch.Generator().manual_seed(2790 + L)
    fi = torch.randperm(len(x), generator=g)[:a.frows]
    fh = (win[fi] % 10) == 0
    fdata = (x[fi[~fh]].cuda(), x[fi[fh]].cuda()) if a.gamma else None
    od = f"{OUT}/band/{a.arm}/L{L}/experts"; os.makedirs(od, exist_ok=True)
    for E in (range(255, -1, -1) if a.rev else range(256)):   # --rev: helper worker from the top
        if E % a.nw != a.w or (os.path.exists(f"{od}/E{E}.json") and not a.force):
            continue
        t0 = time.time()
        rows, sl = torch.where(ids == E)
        pe = p[rows, sl]
        off = None
        if Dm is not None:
            off = (Dm[rows].float() * (pe.double() / sp2[rows]).float()[:, None]).bfloat16()
        data = T.ActData.from_tensors(x[rows], pe, win[rows], "cuda", off=off)
        src = f"{OUT}/band/{a.base_arm}/L{L}/experts" if a.base_arm else f"{a.enc}/L{L}/experts"
        art = torch.load(f"{src}/E{E}.pt", weights_only=False, map_location="cpu")
        teacher = [w.to("cuda", torch.float32) for w in weights(SRC, L, E)]
        print(f"[L{L} E{E}] {a.target} train {data.n_train} held {data.n_hold}", flush=True)
        M, info = T.tune(art, teacher, data, objective="act", a=a.a, steps=a.steps, lr=a.lr, batch=a.batch,
                         eval_every=a.eval_every, patience=a.patience, norm=True, warmup=a.warmup, cap_q=a.cap_q,
                         gamma=a.gamma, fdata=fdata, log=(lambda m: print(m, flush=True)) if a.verbose else (lambda m: None))
        new = M.write(art); T.layout_check(art, new)
        torch.save(new, f"{od}/E{E}.pt.tmp"); os.replace(f"{od}/E{E}.pt.tmp", f"{od}/E{E}.pt")
        re = torch.load(f"{od}/E{E}.pt", weights_only=False, map_location="cpu"); T.layout_check(art, re)
        det = {}
        with torch.no_grad():
            for l in (2, 4):
                d1 = D_.decode_expert(re, l, "cuda"); d2 = D_.decode_expert(re, l, "cuda")
                det[l] = all(torch.equal(u, v) for u, v in zip(d1, d2)); del d1, d2
        info.update(layer=L, expert=E, arm=a.arm, base_arm=a.base_arm, target=a.target, dump=a.dump, a=a.a, lr=a.lr, steps=a.steps,
                    batch=a.batch, gamma=a.gamma, cap_q=a.cap_q, tf32=a.tf32, deterministic=det, layout_identical=True,
                    total_s=time.time() - t0)
        info.pop("hist", None)
        json.dump(info, open(f"{od}/E{E}.json", "w"), indent=1)
        h0, hb = info["held0"], info["held_best"]
        print(f"[L{L} E{E}] L2 {100*h0['2']:.3f}->{100*hb['2']:.3f} L4 {100*h0['4']:.3f}->{100*hb['4']:.3f} "
              + (f"F2 {100*h0['f2']:.3f}->{100*hb['f2']:.3f} F4 {100*h0['f4']:.3f}->{100*hb['f4']:.3f} " if a.gamma else "")
              + f"best@{info['best_step']} det {det} ({info['total_s']:.0f}s)", flush=True)
        del M, data, teacher, art, new, re
        torch.cuda.empty_cache()


def cmd_eval(a):
    """Layer-level held-out (window % 10 == 0) routed-MoE output error of the DEPLOYED nqdef mix (defset L4 experts,
    rest L2), base (nq-encode-v1) vs each arm, against both targets:
      same: R_fp8 = sum_k p_k f^FP8_{E_k}(x_nqdef)            ref: d_ref - shared (FP8's own layer output, routed part)
    also pure-L2 / pure-L4 layer errors.  All SwiGLU math bf16 (teacher_bf16) on decoded fp16 weights."""
    import nq_decode as D_
    import nq27_tune as T
    from orbit_duet.source import weights
    torch.cuda.set_per_process_memory_fraction(a.gpu_gb / 80)
    L = a.layer
    l4 = set(json.load(open("/tmp/nestquant/18-e2e/defset.json"))["layers"][str(L)])
    win = windows(a.dump); hm = (win % 10) == 0
    hi = torch.where(hm)[0]
    x = load(a.dump, L, "x_nqdef")[hi]; ids = load(a.dump, L, "ids_nqdef")[hi]; p = load(a.dump, L, "p_nqdef")[hi]
    sh = load(a.dump, L, "shared_nqdef")[hi].float(); dref = load(a.dump, L, "d_ref_nqdef")[hi].float() - sh; del sh
    arms = ["base"] + [s for s in a.arms.split(",") if s]
    srcs = {"base": f"{a.enc}/L{L}/experts"} | {m: f"{OUT}/band/{m}/L{L}/experts" for m in arms[1:]}
    outs = {(m, lv): torch.zeros_like(dref) for m in arms for lv in ("mix", 2, 4)}
    R = torch.zeros_like(dref)
    xg = x.cuda()
    for E in range(256):
        rows, sl = torch.where(ids == E)
        if len(rows) == 0:
            continue
        xr = xg[rows.cuda()]; pr = p[rows, sl].cuda()[:, None]
        tb = [w.to("cuda").bfloat16() for w in weights(SRC, L, E)]
        R.index_add_(0, rows, (T.teacher_bf16(xr, tb) * pr).cpu())
        for m in arms:
            art = torch.load(f"{srcs[m]}/E{E}.pt", weights_only=False, map_location="cpu")
            ys = {}
            for lv in (2, 4):
                W = [w.to("cuda").bfloat16() for w in D_.decode_expert(art, lv, "cuda")]
                ys[lv] = (T.teacher_bf16(xr, W) * pr).cpu(); del W
            ys["mix"] = ys[4 if E in l4 else 2]
            for lv in ("mix", 2, 4):
                outs[(m, lv)].index_add_(0, rows, ys[lv])
    res = dict(layer=L, dump=a.dump, n_tokens=len(hi), l4_experts=len(l4),
               rel_drift_held=float((dref - R).norm() / dref.norm()))
    for m in arms:
        for lv in ("mix", 2, 4):
            o = outs[(m, lv)]
            res[f"{m}/{lv}/same"] = float((o - R).norm() / R.norm())
            res[f"{m}/{lv}/ref"] = float((o - dref).norm() / dref.norm())
    os.makedirs(f"{OUT}/band/eval", exist_ok=True)
    tag = "_".join(arms[1:]) or "base"
    json.dump(res, open(f"{OUT}/band/eval/L{L}_{tag_of(a.dump)}_{tag}.json", "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)


def cmd_down(a):
    """Downstream effect of a band override: per dumped layer, nqdef-stream vs FP8 (FP8 always from the base dump,
    same token order): relative hidden-state error ||h_out - h_out_fp8|| / ||h_out_fp8||, relative MoE-output error,
    router agreement (mean |top8 & top8_fp8| / 8) and top-1 agreement; all tokens and held-out windows only."""
    win = windows(a.base); hm = (win % 10) == 0
    res = {}
    for L in map(int, a.layers.split(",")):
        r = {}
        f8 = {k: load(a.base, L, f"{k}_fp8") for k in ("h_out", "moe_out", "ids")}
        for tag, d in (("base", a.base), ("ovr", a.dump)):
            if not os.path.exists(f"{d}/L{L}"):
                continue
            s = {k: load(d, L, f"{k}_nqdef") for k in ("h_out", "moe_out", "ids")}
            for sub, m in (("all", slice(None)), ("held", hm)):
                e = lambda k: float((s[k][m].float() - f8[k][m].float()).norm() / f8[k][m].float().norm())
                i1, i2 = s["ids"][m].long(), f8["ids"][m].long()
                ov = (torch.zeros(len(i1), 256, dtype=torch.bool).scatter_(1, i1, True)
                      & torch.zeros(len(i2), 256, dtype=torch.bool).scatter_(1, i2, True)).sum(1).float() / 8
                med = lambda k: float(((s[k][m].float() - f8[k][m].float()).norm(dim=1)
                                       / f8[k][m].float().norm(dim=1).clamp_min(1e-12)).median())
                r[f"{tag}/{sub}"] = dict(h_out=e("h_out"), moe_out=e("moe_out"), h_out_med=med("h_out"),
                                         moe_out_med=med("moe_out"), router_overlap=float(ov.mean()),
                                         top1=float((i1[:, 0] == i2[:, 0]).float().mean()))
            del s
        res[L] = r
        print(L, json.dumps(r), flush=True)
    os.makedirs(f"{OUT}/band/eval", exist_ok=True)
    json.dump(res, open(f"{OUT}/band/eval/down_{tag_of(a.dump)}.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    dn = sub.add_parser("down"); dn.add_argument("--dump", required=True); dn.add_argument("--base", default=DUMP)
    dn.add_argument("--layers", default="30,31,32")
    ev = sub.add_parser("eval"); ev.add_argument("--layer", type=int, required=True); ev.add_argument("--dump", default=DUMP)
    ev.add_argument("--enc", default=ENC); ev.add_argument("--arms", default=""); ev.add_argument("--gpu-gb", type=float, default=10.5)
    pr = sub.add_parser("prep"); pr.add_argument("--layer", type=int, required=True); pr.add_argument("--dump", default=DUMP)
    pr.add_argument("--gpu-gb", type=float, default=10.5)
    t = sub.add_parser("tune")
    t.add_argument("--layer", type=int, required=True); t.add_argument("--target", choices=["same", "ref"], required=True)
    t.add_argument("--arm", required=True); t.add_argument("--dump", default=DUMP)
    t.add_argument("--w", type=int, default=0); t.add_argument("--nw", type=int, default=1)
    t.add_argument("--a", type=float, default=0.5); t.add_argument("--steps", type=int, default=800)
    t.add_argument("--lr", type=float, default=3e-3); t.add_argument("--batch", type=int, default=2048)
    t.add_argument("--eval-every", type=int, default=50); t.add_argument("--patience", type=int, default=6)
    t.add_argument("--warmup", type=int, default=50); t.add_argument("--cap-q", type=float, default=0.99)
    t.add_argument("--gamma", type=float, default=1.0); t.add_argument("--frows", type=int, default=32768)
    t.add_argument("--tf32", action="store_true"); t.add_argument("--gpu-gb", type=float, default=10.5)
    t.add_argument("--force", action="store_true"); t.add_argument("--verbose", action="store_true")
    t.add_argument("--rev", action="store_true", help="descending expert order (extra worker on a partly-done arm)")
    t.add_argument("--enc", default=ENC, help="untuned base encode (nq-encode-v1; L3-6: T29's nq-encode-h512)")
    t.add_argument("--base-arm", default="", help="start from OUT/band/ARM (e.g. a nq27_pv.py re-encode) not nq-encode-v1")
    a = ap.parse_args()
    {"prep": cmd_prep, "tune": cmd_tune, "eval": cmd_eval, "down": cmd_down}[a.cmd](a)


if __name__ == "__main__":
    main()
