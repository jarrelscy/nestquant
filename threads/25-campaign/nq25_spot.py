"""Thread 25 spot check: 2 experts per layer (one from fixed_set.json, one ordinary), nq L2/L4 vs EXL3-2/EXL3-4.

  python nq25_spot.py --layer L --stats SHIM --out ROOT [--experts a,b] [--l2-thr 5] [--l4-thr 2]

Same H and eval rows for all arms, as thread 12's nq_dist48.py / nq_diag48.py run them:
  H      = T12 nq_layer.expert_HG(open_stats(SHIM[, VISION_SHIM, w]), L, E)   (exactly the encode's H: pinned stats,
           text/vision blend if configured, unrouted fallbacks H=I)
  EXL3-K = harness.quantize_exl3_like(teacher, H, K, count=1, sigma_reg=sigma[proj])   (K = 2, 4)
  nq     = nq_decode.decode_expert(ROOT/L{L}/experts/E{E}.pt, level)   (the campaign artifact itself, no re-encode)
  eval   = harness.table(harness.evaluate(cap.expert_data(L, E, "matched"), methods, groups=True))
Selection (deterministic): fixed = fixed-set member with the most matched routed rows among those with >= MIN_ROWS
(else most rows); ordinary = a seeded (seed L) random non-fixed expert with >= MIN_ROWS rows (else most rows).
Flag: nq/L2 - EXL3-2 > l2-thr % or nq/L4 - EXL3-4 > l4-thr % (relative, on all/routed; on all/forced for an expert
with < MIN_ROWS routed eval rows, where routed deltas are noise: dry run L19 E0 had 29 rows, +29% routed / +0.7% forced).
Writes ROOT/spot/L{L}.json.
"""
import os, sys, json, time, argparse
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np
import torch
T12 = "/home/coder/git/nestquant/threads/12-reference-encoder"
T19 = "/home/coder/git/nestquant/threads/19-full-capture"
for p in (T12, T19):
    sys.path.insert(0, p)
MIN_ROWS = 256
PROJ = ("gate", "up", "down")
SIGMA = {"gate": 0.5, "up": 0.5, "down": 1.0}          # thread-08 selection (nq19_load FORMAT.md)
KEYS = ("all/routed", "all/forced", "ood/routed", "ood/forced", "control/routed")


def bpw(D, art):
    """expert bits/weight at L2 / L4 (nq_decode.bits_per_level, weighted by k*n over gate/up/down)."""
    try:
        per = {pn: D.bits_per_level(art[pn]) for pn in PROJ}
        nw = {pn: art[pn]["meta"]["k"] * art[pn]["meta"]["n"] for pn in PROJ}
        return {lv: sum(per[pn][lv] * nw[pn] for pn in PROJ) / sum(nw.values()) for lv in (2, 4)}
    except Exception as e:
        return str(e)[:80]


def rel(a, b):
    return None if a is None or b is None else 100 * (a / b - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", default="/tmp/nestquant/src/glm53-fp8")
    ap.add_argument("--fixed-set", default="/home/coder/git/nestquant/threads/22-boundary-experts/fixed_set.json")
    ap.add_argument("--experts", help="override selection: E_fixed,E_ordinary")
    ap.add_argument("--l2-thr", type=float, default=5.0)
    ap.add_argument("--l4-thr", type=float, default=2.0)
    ap.add_argument("--key", default="all/routed")
    ap.add_argument("--h-fn", help="override: module:function(cap, L, E, vision_stats, w) -> HG; default T12 expert_HG")
    ap.add_argument("--vision-stats"); ap.add_argument("--vision-weight", type=float, default=0.0)
    ap.add_argument("--eval", default="matched", help="eval set: matched (production spot) | val (held-out, ~13x rows)")
    ap.add_argument("--tag", default="", help="output ROOT/spot/L{L}{tag}.json")
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(12 / 80)
    torch.backends.cuda.matmul.allow_tf32 = False
    import nq19_load, harness as h, nq_decode as D
    try:
        import nq_encode as NE
        sig = dict(NE.PROD["sigma"])
    except Exception:
        sig = SIGMA
    hfn = None
    if a.h_fn:
        import importlib
        mod, fn = a.h_fn.split(":")
        hfn = getattr(importlib.import_module(mod), fn)
    import nq_layer as NL
    L = a.layer
    cap = nq19_load.Capture(root=a.stats)
    hcap = NL.open_stats(a.stats, a.vision_stats, a.vision_weight) if a.vision_stats else cap
    fixed = [int(e) for e in json.load(open(a.fixed_set))["fixed_set"][str(L)]]
    ids = torch.load(cap.eval_path(L, a.eval), weights_only=True, mmap=True)["ids"]
    rows = np.bincount(ids.flatten().numpy(), minlength=256)
    if a.experts:
        ef, eo = map(int, a.experts.split(","))
    else:
        ok_f = [e for e in fixed if rows[e] >= MIN_ROWS]
        ef = max(ok_f or fixed, key=lambda e: rows[e])
        rest = [e for e in range(256) if e not in fixed]
        ok_o = [e for e in rest if rows[e] >= MIN_ROWS]
        eo = int(np.random.default_rng(L).choice(ok_o)) if ok_o else max(rest, key=lambda e: rows[e])
    res = dict(layer=L, stats=os.path.realpath(f"{a.stats}/stats/L{L}"), h_fn=a.h_fn, vision_weight=a.vision_weight,
               vision_stats=os.path.realpath(f"{a.vision_stats}/stats/L{L}") if a.vision_stats else None, key=a.key, thr=dict(L2=a.l2_thr, L4=a.l4_thr),
               experts={}, time=time.strftime("%Y-%m-%d %H:%M:%S"))
    flag = False
    for role, E in (("fixed", ef), ("ordinary", eo)):
        t0 = time.time()
        art = torch.load(f"{a.out}/L{L}/experts/E{E}.pt", weights_only=False, map_location="cpu")
        dm = cap.expert_data(L, E, a.eval, source=a.source)
        HG = hfn(cap, L, E, a.vision_stats, a.vision_weight) if hfn else NL.expert_HG(hcap, L, E)[0]   # encode's H
        methods = {}
        for K in (2, 4):
            q = []
            for pi, pn in enumerate(PROJ):
                Hm = HG["H"][pi]
                Wq, _ = h.quantize_exl3_like(dm.teacher[pi], Hm, K, count=1, sigma_reg=sig[pn])
                q.append(Wq.cpu()); h.free_scratch()
            methods[f"EXL3-{K}"] = q
        for Lv in (2, 4):
            methods[f"nq/L{Lv}"] = [w.cpu() for w in D.decode_expert(art, Lv)]
        ev = {}
        names = list(methods)
        for j in range(0, len(names), 2):
            grp = {n: [w.cuda() for w in methods[n]] for n in names[j:j + 2]}
            tb = h.table(h.evaluate(dm, grp, groups=True))
            ev.update({n: tb[n] for n in grp})
            del grp; torch.cuda.empty_cache()
        key = a.key if rows[E] >= MIN_ROWS or "routed" not in a.key else a.key.replace("routed", "forced")
        d2 = rel(ev["nq/L2"].get(key), ev["EXL3-2"].get(key))       # < MIN_ROWS routed rows: judged on forced rows
        d4 = rel(ev["nq/L4"].get(key), ev["EXL3-4"].get(key))
        f = (d2 is not None and d2 > a.l2_thr) or (d4 is not None and d4 > a.l4_thr)
        flag |= f
        res["experts"][role] = dict(expert=E, routed_rows=int(rows[E]), flag_key=key, eval=ev, flag=f,
                                    delta_pct={k: dict(L2=rel(ev["nq/L2"].get(k), ev["EXL3-2"].get(k)),
                                                       L4=rel(ev["nq/L4"].get(k), ev["EXL3-4"].get(k)))
                                               for k in KEYS if k in ev["EXL3-2"]},
                                    rate=art["meta"].get("rate"), flags=art["meta"].get("flags"), bpw=bpw(D, art),
                                    s=round(time.time() - t0))
        print(f"[L{L} E{E} {role}] rows {rows[E]} {key} dL2 {d2:+.2f}% dL4 {d4:+.2f}% flag {f} ({time.time()-t0:.0f}s)", flush=True)
        del art, dm, HG, methods; torch.cuda.empty_cache()
    res["flag"] = flag
    res["summary"] = {r: f"E{v['expert']} ({v['flag_key']}, {v['routed_rows']} rows) L2 {v['delta_pct'][v['flag_key']]['L2']:+.2f}% "
                         f"L4 {v['delta_pct'][v['flag_key']]['L4']:+.2f}%"
                      for r, v in res["experts"].items()}
    os.makedirs(f"{a.out}/spot", exist_ok=True)
    res["eval_set"] = a.eval
    p = f"{a.out}/spot/L{L}{a.tag}.json"
    json.dump(res, open(p + ".tmp", "w"), indent=1); os.replace(p + ".tmp", p)
    print(json.dumps(res["summary"]), "flag", flag, flush=True)


if __name__ == "__main__":
    main()
