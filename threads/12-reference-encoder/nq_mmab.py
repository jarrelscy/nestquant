"""Vision-blend weight A/B: production encoder (lr on) with H from w = 0 / 0.1 / 0.25 (nq26_blend via nq_layer.open_stats).

  CUDA_VISIBLE_DEVICES=1 python nq_mmab.py select            # -> results_mmab/selection.json
  CUDA_VISIBLE_DEVICES=1 python nq_mmab.py run --part 0/2

Eval (rel L2 %, harness): text matched (all/routed, ood/forced), text val (all), vision val (19-capture-mm eval/val).
Results -> results_mmab/L{L}_E{E}.json (eval[split][arm/Lv]).
"""
import os, sys, json, time, argparse
os.environ.setdefault("OMP_NUM_THREADS", "16")
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nq_layer as NL
import nq_encode as NE

TEXT = "/tmp/nestquant/19-capture-glmfmt"
MM = "/tmp/nestquant/19-capture-mm"
OUT = f"{HERE}/results_mmab"
WS = (0.0, 0.1, 0.25)
LAYERS = (10, 30, 50, 70)


def select():
    sys.path.insert(0, NL.T19)
    import nq19_load
    cv = nq19_load.Capture(root=MM)
    sel = {}
    for L in LAYERS:
        nv = np.asarray(cv._arr(L, "scalars"))[:, 0].astype(np.float64)
        o = np.argsort(-nv, kind="stable")
        hi = int(o[0])                                                 # highest vision usage
        mid = int(o[len(o) // 4])                                      # ordinary (upper quartile of usage)
        part = [int(e) for e in o if 0 < nv[e] < 128]
        low = part[len(part) // 2] if part else int(o[len(o) // 2])    # partial weight (n_v < 128)
        sel[L] = {"hi_vision": dict(expert=hi, n_v=int(nv[hi])), "ordinary": dict(expert=mid, n_v=int(nv[mid])),
                  "low_vision": dict(expert=low, n_v=int(nv[low]))}
    os.makedirs(OUT, exist_ok=True)
    json.dump(sel, open(f"{OUT}/selection.json", "w"), indent=1)
    print(json.dumps(sel))


def run(part):
    import harness as h
    i, m = map(int, part.split("/"))
    torch.cuda.set_per_process_memory_fraction(12 / 80)
    torch.backends.cuda.matmul.allow_tf32 = False
    caps = {w: NL.open_stats(TEXT, MM if w else None, w) for w in WS}
    sel = json.load(open(f"{OUT}/selection.json"))
    jobs = [(int(L), r, v) for L, d in sel.items() for r, v in d.items()]
    for L, role, v in jobs[i::m]:
        E = v["expert"]
        path = f"{OUT}/L{L}_E{E}.json"
        R = json.load(open(path)) if os.path.exists(path) else dict(layer=L, expert=E, role=role, n_v=v["n_v"], eval={}, meta={})
        t0 = time.time()
        dm = caps[0.0].expert_data(L, E, "matched")
        methods = {}
        for w in WS:
            tag = f"w{w:g}"
            if f"{tag}/L4" in R["eval"].get("vision_val", {}):
                continue
            HG, flags = NL.expert_HG(caps[w], L, E)
            t1 = time.time()
            art, dense = NE.encode_expert(dm.teacher, HG)            # production defaults (lr on)
            R["meta"][tag] = dict(flags=flags, w_eff=HG["meta"].get("w_vision_eff", 0.0), rate=art["meta"]["rate"],
                                  lr_rank=art["meta"]["lr_rank"], encode_s=time.time() - t1,
                                  bitexact={p: art["meta"]["info"][p]["bitexact"] for p in NE.PROJ})
            for Lv in (2, 4):
                methods[f"{tag}/L{Lv}"] = [x.cpu() for x in dense[Lv]]
            del art, dense, HG; torch.cuda.empty_cache()
            print(f"[L{L} E{E}] {tag} enc {R['meta'][tag]['encode_s']:.0f}s w_eff {R['meta'][tag]['w_eff']}", flush=True)
        if methods:
            splits = (("text_matched", dm, True), ("text_val", None, False), ("vision_val", None, False))
            for name, data, groups in splits:
                if data is None:
                    if name == "text_val":
                        data = caps[0.0].expert_data(L, E, "val")
                    else:                                   # T26 loader: pad rows dropped, image + text rows, groups img/txt
                        sys.path.insert(0, NL.T26)
                        import nq26_eval
                        data = nq26_eval.expert_data(L, E, rows="valid"); groups = True
                names = list(methods)
                for j in range(0, len(names), 3):
                    grp = {n_: [x.cuda() for x in methods[n_]] for n_ in names[j:j + 3]}
                    ev_ = h.evaluate(data, grp, groups=groups)
                    tb = h.table(ev_, domains=[d for d in ev_ if isinstance(ev_[d], dict) and "forced" in ev_[d]])
                    R["eval"].setdefault(name, {}).update({n_: tb[n_] for n_ in grp})
                    del grp; torch.cuda.empty_cache()
                if name != "text_matched":
                    del data
        json.dump(R, open(path + ".tmp", "w"), indent=1); os.replace(path + ".tmp", path)
        print(f"[L{L} E{E} {role}] done {time.time()-t0:.0f}s", flush=True)
        del dm, methods; torch.cuda.empty_cache()


def summary():
    import glob
    rows = [json.load(open(f)) for f in sorted(glob.glob(f"{OUT}/L*_E*.json"))]
    keys = (("text_matched", "all/routed"), ("text_matched", "ood/forced"), ("text_val", "all/routed"),
            ("text_val", "all/forced"), ("vision_val", "all/routed"), ("vision_val", "all/forced"),
            ("vision_val", "img/routed"), ("vision_val", "txt/routed"))
    for Lv in (2, 4):
        print(f"== L{Lv}: rel L2 %, and delta % vs w0 per expert")
        agg = {(s, k, w): [] for s, k in keys for w in WS}
        for R in rows:
            line = f"L{R['layer']:2d}E{R['expert']:<3d} {R['role'][:8]:8s} nv{R['n_v']:<6d}"
            for s, k in keys:
                ev = R["eval"].get(s, {})
                b = (ev.get(f"w0/L{Lv}") or {}).get(k)
                for w in WS:
                    x = (ev.get(f"w{w:g}/L{Lv}") or {}).get(k)
                    if x is None or b is None:
                        continue
                    agg[(s, k, w)].append(100 * (x / b - 1))
                    if w:
                        line += f" {100*(x/b-1):+6.2f}"
                    else:
                        line += f" |{x:7.3f}"
            print(line)
        for s, k in keys:
            print(f"  {s:12s} {k:10s} " + "  ".join(f"w{w:g} mean {np.mean(agg[(s,k,w)]):+.2f} worst {np.max(agg[(s,k,w)]):+.2f} (n {len(agg[(s,k,w)])})"
                                               for w in WS[1:]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("select", "run", "summary"))
    ap.add_argument("--part", default="0/1")
    a = ap.parse_args()
    {"select": select, "run": lambda: run(a.part), "summary": summary}[a.cmd]()
