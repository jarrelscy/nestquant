"""T27 scoring on eval/val (the thread-25 spot rows, never used by the tuner).

  python nq27_eval.py --arms A,B L:E [L:E ...]

Per expert, exactly the nq25_spot.py --eval val arithmetic: H = T12 nq_layer.expert_HG(open_stats(_stats, _stats_mm,
0.25), L, E) (the encode's H), EXL3-K = harness.quantize_exl3_like(teacher, H, K, count=1, sigma_reg=PROD sigma),
nq = nq_decode.decode_expert(artifact, level), eval = harness.table(harness.evaluate(cap.expert_data(L, E, "val"))).
EXL3 + untuned nq (the campaign artifact) are cached in OUT/val/base/L{L}_E{E}.json; each arm's tuned artifact
OUT/arms/ARM/L{L}/experts/E{E}.pt -> OUT/val/ARM/L{L}_E{E}.json.
"""
import os, sys, json, time, argparse
os.environ.setdefault("OMP_NUM_THREADS", "8")
import torch

SRC = "/tmp/nestquant/src/glm53-fp8"
R = "/tmp/nestquant/nq-encode-v1"
OUT = "/tmp/nestquant/27-pv-tune"
PROJ = ("gate", "up", "down")


def evaluate_all(h, data, methods):
    """harness.evaluate restricted to the 'all' domain (identical arithmetic: harness._errors on the same rows)."""
    cap = data.capture
    routed, slots = torch.where(cap["ids"] == data.expert)
    rows = torch.arange(len(cap["x"]))
    return {"all": dict(forced=h._errors(cap, data.teacher, methods, rows, torch.ones(len(rows))),
                        routed=h._errors(cap, data.teacher, methods, routed, cap["p"][routed, slots]))}


def ev_methods(h, dm, methods):
    ev = {}; names = list(methods)
    for j in range(0, len(names), 2):
        grp = {n: [w.cuda() for w in methods[n]] for n in names[j:j + 2]}
        tb = h.table(evaluate_all(h, dm, grp), domains=("all",))
        ev.update({n: tb[n] for n in grp})
        del grp; torch.cuda.empty_cache()
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=""); ap.add_argument("--gpu-gb", type=float, default=12)
    ap.add_argument("--loop", action="store_true", help="arms only: skip pairs without a base yet, repeat until all done")
    ap.add_argument("pairs", nargs="+")
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(a.gpu_gb / 80)
    torch.backends.cuda.matmul.allow_tf32 = False
    import nq19_load, harness as h, nq_decode as D, nq_layer as NL, nq_encode as NE
    sig = dict(NE.PROD["sigma"])
    cap = nq19_load.Capture(root=f"{R}/_stats")
    hcap = NL.open_stats(f"{R}/_stats", f"{R}/_stats_mm", 0.25)
    arms = [x for x in a.arms.split(",") if x]
    pending = list(a.pairs)
    while pending:
        pending = run_pairs(a, pending, arms, cap, hcap, sig, h, D, NL) if a.loop else run_pairs(a, pending, arms, cap, hcap, sig, h, D, NL) and []
        if pending:
            time.sleep(120)


def run_pairs(a, pairs, arms, cap, hcap, sig, h, D, NL):
    left = []
    for pr in pairs:
        L, E = map(int, pr.split(":"))
        bp = f"{OUT}/val/base/L{L}_E{E}.json"
        if a.loop:
            want = [x for x in arms if not os.path.exists(f"{OUT}/val/{x}/L{L}_E{E}.json")]
            ready = [x for x in want if os.path.exists(f"{OUT}/arms/{x}/L{L}/experts/E{E}.json")]
            if want:
                left.append(pr)
            if not os.path.exists(bp) or not ready:
                continue
        todo = [x for x in arms if not os.path.exists(f"{OUT}/val/{x}/L{L}_E{E}.json")
                and os.path.exists(f"{OUT}/arms/{x}/L{L}/experts/E{E}.pt")]
        if os.path.exists(bp) and not todo:
            continue
        t0 = time.time()
        dm = cap.expert_data(L, E, "val", source=SRC)
        if not os.path.exists(bp):
            HG = NL.expert_HG(hcap, L, E)[0]
            methods = {}
            for K in (2, 4):
                q = []
                for pi, pn in enumerate(PROJ):
                    Wq, _ = h.quantize_exl3_like(dm.teacher[pi], HG["H"][pi], K, count=1, sigma_reg=sig[pn])
                    q.append(Wq.cpu()); h.free_scratch()
                methods[f"EXL3-{K}"] = q
            del HG
            art = torch.load(f"{R}/L{L}/experts/E{E}.pt", weights_only=False, map_location="cpu")
            for Lv in (2, 4):
                methods[f"nq/L{Lv}"] = [w.cpu() for w in D.decode_expert(art, Lv)]
            ev = ev_methods(h, dm, methods)
            os.makedirs(os.path.dirname(bp), exist_ok=True)
            json.dump(dict(layer=L, expert=E, eval=ev, s=round(time.time() - t0)), open(bp, "w"), indent=1)
            del methods, art
            print(f"L{L} E{E} base {ev} ({time.time()-t0:.0f}s)", flush=True)
        for x in todo:
            art = torch.load(f"{OUT}/arms/{x}/L{L}/experts/E{E}.pt", weights_only=False, map_location="cpu")
            methods = {f"L{Lv}": [w.cpu() for w in D.decode_expert(art, Lv)] for Lv in (2, 4)}
            ev = ev_methods(h, dm, methods)
            p = f"{OUT}/val/{x}/L{L}_E{E}.json"; os.makedirs(os.path.dirname(p), exist_ok=True)
            json.dump(dict(layer=L, expert=E, arm=x, eval=ev), open(p, "w"), indent=1)
            print(f"L{L} E{E} {x} {ev}", flush=True)
            del art, methods
        del dm; torch.cuda.empty_cache()
    return left


if __name__ == "__main__":
    main()
