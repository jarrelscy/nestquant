"""T29 validator wrapper for a serving release with in_had_down (thread 28 nq_check.py + the rotation it cannot see).

nq_check (c) compares ROTATED-domain levels (published planes vs nq_decode ring levels), so it passes whatever the input
Hadamard is.  This wrapper adds, for the selected layers:
  (m) metadata: serving manifest.json in_had_down[L] == layers/L{L}.json source.in_had_down == reference layer manifest
      config.in_had_down (absent = 128), for every layer present
  (k) kernel-semantics forward: the moe.Expert.ref math on the PUBLISHED planes (resident base/var/scales/lr + record
      P4/d4/U4, per rank) with the down-input WHT at in_had_down, vs the dense reference forward through decode29
      (nq29_had.decode_expert29 of the reference layer, rank slice); rel err must be < --tol at levels 2 and 4, and for
      in_had_down != 128 the Had128 reading must FAIL (> 100x tol: the field is load-bearing)
  python check29.py DIR [--ref ROOT] [--layers 3-6] [--experts 3] [--no-nqcheck] [--testvec OUTDIR L:E:rank]
Prints 'T29 CHECK PASS' / 'T29 CHECK FAIL'."""
import os, sys, json, time, random, argparse, subprocess
import torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
T28 = "/home/coder/git/nestquant/threads/28-serve-release"
sys.path.insert(0, T28)
import nq_release as NR
import nq_check as NC
import nq29_had as NH
BAD = []


def bad(m):
    BAD.append(m); print("  FAIL " + m, flush=True)


def sylv(n, dev):
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return (H / n ** 0.5).to(dev)


def fwd_pub(pg, pd, sc, lr, lr4, rg, rd, x, level, w_down):
    """moe.Expert.ref on published planes; down-input WHT at width w_down.  -> dict(y, h, hrot)"""
    from moe import dense_W
    H, I = pd.N, pd.K; dev = x.device
    H128 = sylv(128, dev); Hw = sylv(w_down, dev)
    wht = lambda v, M: (v.view(*v.shape[:-1], -1, M.shape[0]) @ M).view(v.shape)
    sg = sc.float().to(dev)
    su, svg, svu, sud, svo, suu = sg[:H], sg[H:H + I], sg[H + I:H + 2 * I], sg[H + 2 * I:H + 3 * I], sg[H + 3 * I:2 * H + 3 * I], sg[2 * H + 3 * I:]
    Wg = dense_W(pg, level, 4).float(); Wd = dense_W(pd, level, 4).float()
    xg = wht(x * su, H128).half().float(); xu = wht(x * suu, H128).half().float()
    a = torch.cat([xg @ Wg[:I].T, xu @ Wg[I:].T], 1)
    g = wht(a[:, :I], H128) * svg; u = wht(a[:, I:], H128) * svu
    if rg + rd:
        f = lr.float().to(dev); f4 = lr4.float().to(dev); o = 0
        def take(t, n, m):
            nonlocal o
            v = t[o:o + n * m].view(n, m); o += n * m; return v
        Vg = take(f, rg, H); U2g = take(f, rg, I); U2u = take(f, rg, I); Vd = take(f, rd, I); U2d = take(f, rd, H)
        o = 0; U4g = take(f4, rg, I); U4u = take(f4, rg, I); U4d = take(f4, rd, H)
        z = x @ Vg.T
        g = g + z @ U2g + (z @ U4g if level == 4 else 0); u = u + z @ U2u + (z @ U4u if level == 4 else 0)
    sw = torch.nn.functional.silu(g) * u
    h = sw * sud
    hrot = wht(h, Hw).half().float()
    y = wht(hrot @ Wd.T, H128) * svo
    if rg + rd:
        z = sw @ Vd.T; y = y + z @ U2d + (z @ U4d if level == 4 else 0)
    return dict(y=y, h=h, hrot=hrot, sw=sw)


def fwd_ref(W, x, r, I):
    Wg, Wu, Wd = (t.float() for t in W)
    s = slice(r * I, (r + 1) * I)
    return (torch.nn.functional.silu(x @ Wg[s].T) * (x @ Wu[s].T)) @ Wd[:, s].T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir"); ap.add_argument("--ref"); ap.add_argument("--layers", default="3-6")
    ap.add_argument("--experts", type=int, default=3); ap.add_argument("--tol", type=float, default=2e-3)
    ap.add_argument("--no-nqcheck", action="store_true"); ap.add_argument("--testvec", nargs=2, metavar=("OUT", "L:E:rank"))
    ap.add_argument("--tokens", type=int, default=16)
    a = ap.parse_args(); T = os.path.abspath(a.dir); ref = a.ref or os.path.dirname(os.path.dirname(T))
    Ls = NR.parse_layers(a.layers)
    if not a.no_nqcheck and not a.testvec:
        cmd = [sys.executable, f"{T28}/nq_check.py", T, "--ref", ref, "--layers", a.layers, "--ranks", "all", "--dev", "cuda",
               "--experts", str(a.experts)]
        print("+", " ".join(cmd), flush=True)
        if subprocess.call(cmd) != 0:
            bad("nq_check.py FAIL")
    torch.cuda.set_per_process_memory_fraction(12 / 80)
    torch.backends.cuda.matmul.allow_tf32 = False
    NQ, PR, RS = NR._imports()
    man = json.load(open(f"{T}/manifest.json")); tp = man["tp"]; lay = man["layout"]; rb = lay["rec_bytes"]
    seg = {k: tuple(v) for k, v in lay["seg"].items()}; H, I = lay["H"], lay["I"]
    # (m) metadata, every layer present
    for L in man["layers_present"]:
        blk = json.load(open(f"{T}/layers/L{L}.json"))
        _, rman, _ = NR.layer_src(ref, L)
        want = int(rman["config"].get("in_had_down", 128))
        got_m = int(man.get("in_had_down", {}).get(str(L), 128)); got_b = int(blk["source"].get("in_had_down", 128))
        if not (want == got_m == got_b):
            bad(f"L{L}: (m) in_had_down reference {want} / serving manifest {got_m} / layer block {got_b}")
    print(f"(m) in_had_down metadata checked on {len(man['layers_present'])} layers "
          f"({sum(int(v) != 128 for v in man.get('in_had_down', {}).values())} non-128)", flush=True)
    # (k) kernel-semantics forward
    todo = []
    if a.testvec:
        L, E, r = map(int, a.testvec[1].split(":")); todo = [(L, [E], [r])]
    else:
        for L in Ls:
            rng = random.Random(2900 + L)
            todo.append((L, sorted(rng.sample(range(256), a.experts)), list(range(tp))))
    g = torch.Generator().manual_seed(29)
    x = torch.randn(a.tokens, H, generator=g).cuda()
    t0 = time.time(); n = 0
    for L, Es, ranks in todo:
        w = int(man.get("in_had_down", {}).get(str(L), 128))
        for r in ranks:
            idx = json.load(open(f"{T}/rank{r}.json")); e = idx["layers"][str(L)]
            res, _, _ = RS.load(f"{T}/{e['res']}", "cpu")
            for E in Es:
                blkb, asm = NC.read_rec(T, idx, L, E, rb); src = blkb if blkb is not None else asm
                xr = res[E]
                pg = NC.published_proj(xr.gu, src, seg, "gu", 2 * I, H, "cuda"); pd = NC.published_proj(xr.dn, src, seg, "dn", H, I, "cuda")
                o, nb = seg["lr4"]; u4 = torch.frombuffer(bytearray(src[o:o + nb]), dtype=torch.float16)
                art = NH.assemble29(ref, L, E) if os.path.exists(f"{ref}/L{L}/tp0.pt") else st_assemble29(ref, L, E)
                out = {}
                for lv in (2, 4):
                    W = NH.decode_expert29(art, lv)
                    yr = fwd_ref(W, x, r, I)
                    pub = fwd_pub(pg, pd, xr.sc[lv], xr.lr, u4, xr.rg, xr.rd, x, lv, w)
                    err = float((pub["y"] - yr).norm() / yr.norm())
                    out[lv] = err
                    if err > a.tol:
                        bad(f"L{L} r{r} E{E}: (k) level-{lv} published forward (had {w}) rel err {err:.2e} > {a.tol}")
                    if w != 128:
                        e128 = float((fwd_pub(pg, pd, xr.sc[lv], xr.lr, u4, xr.rg, xr.rd, x, lv, 128)["y"] - yr).norm() / yr.norm())
                        out[f"{lv}_as128"] = e128
                        if e128 < 100 * a.tol:
                            bad(f"L{L} r{r} E{E}: (k) level-{lv} Had128 reading also passes ({e128:.2e}): field not load-bearing?")
                    if a.testvec:
                        od = a.testvec[0]; os.makedirs(od, exist_ok=True)
                        torch.save(dict(L=L, E=E, rank=r, tp=tp, level=lv, in_had_down=w, x=x.cpu(), sw=pub["sw"].cpu(),
                                        h=pub["h"].cpu(), hrot=pub["hrot"].cpu(), y=pub["y"].cpu(), y_ref_dense=yr.cpu(),
                                        y_as_had128=fwd_pub(pg, pd, xr.sc[lv], xr.lr, u4, xr.rg, xr.rd, x, lv, 128)["y"].cpu(),
                                        note="x [T,H] hidden (random N(0,1), seed 29); sw = silu(g)*u [T,I] rank slice; "
                                             "h = sw*su_d (signs/scales of down input); hrot = WHT_w(h) per 512 block, "
                                             "fp16-rounded (what K2 consumes); y = rank-partial output [T,H] before all-reduce"),
                                   f"{od}/L{L}_E{E}_r{r}_lv{lv}.pt")
                print(f"  L{L} r{r} E{E} had{w}: " + " ".join(f"{k} {v:.2e}" for k, v in out.items()), flush=True)
                n += 1
            del res
    print(f"(k) {n} (layer, rank, expert) forward checks in {time.time()-t0:.0f}s", flush=True)
    print("T29 CHECK", "PASS" if not BAD else "FAIL", flush=True)
    sys.exit(1 if BAD else 0)


def st_assemble29(ref, L, E):
    sys.path.insert(0, "/home/coder/git/nestquant/threads/25-campaign")
    import nq25_st as S
    d = NR._imports()[0].layer_dir(ref, L)
    root, sub = (os.path.dirname(os.path.dirname(d)), True) if d.endswith(f"layers/L{L}") else (os.path.dirname(d), False)
    if sub:                                  # HF layout: nq25_st expects ROOT/L{L}; point it at ROOT/layers
        root = os.path.dirname(d)
    art = S.assemble(root, L, E)
    art["meta"] = {NH.FIELD: int(json.load(open(f"{d}/manifest.json"))["config"].get(NH.FIELD, 128))}
    return art


if __name__ == "__main__":
    main()
