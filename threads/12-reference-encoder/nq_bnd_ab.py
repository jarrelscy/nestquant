"""Boundary-weight A/B on the T19 capture: weight 1 (old) vs 50 (flat), NestQuant L2/L4 and same-H EXL3-2/EXL3-4.

  CUDA_VISIBLE_DEVICES=4 python nq_bnd_ab.py --layer 16 --experts 36,92,165 [--arms 1,50,50/k100,50/cap0.2] [--rate R]

arm = weight spec[/k<K>][/cap<F>]: ESS shrink w_eff = 1+(w-1)ESS/(ESS+K) and/or trace-share cap F (nq_bnd.glm_H_bnd).

Splits: matched (orbit docs: all/control/ood, forced+routed), val (T19 held-out: all, forced+routed) and the boundary
split on val (think/end at d=1 and d<=32; needs the eval capture's bnd_think / bnd_end labels) -> results_bnd/L{L}_E{E}.json
"""
import os, sys, json, time, argparse
os.environ.setdefault("OMP_NUM_THREADS", "16")
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
T19 = "/home/coder/git/nestquant/threads/19-full-capture"
sys.path.insert(0, T19)
import harness as h
import nq_encode as NE
import nq_bnd as NB

PROJ = NE.PROJ
SIG = NE.PROD["sigma"]
RES = os.environ.get("NQ_BND_RES", f"{HERE}/results_bnd")


def tables(data, methods, bnd=True, groups=True):
    out = {}
    names = list(methods)
    for i in range(0, len(names), 4):
        grp = {n: [w.cuda() for w in methods[n]] for n in names[i:i + 4]}
        ev = h.evaluate(data, grp, groups=groups)
        if bnd:
            ev.update(NB.evaluate_boundary(data, grp))
        doms = [d for d in ev if d in ("all", "control", "ood") or d.startswith("bnd:")]
        tb = h.table(ev, domains=doms)
        rows = {d: {k: (ev[d][k] or {}).get("rows") for k in ("forced", "routed")} for d in doms}
        for n in grp:
            out[n] = tb[n]
        out["_rows"] = rows
        del grp; torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--experts", default="36,92,165")
    ap.add_argument("--arms", "--weights", dest="arms", default="1,50",
                    help="comma list: flat weight[/k<K>][/cap<F>]")
    ap.add_argument("--canon", type=int, default=0)
    ap.add_argument("--inner", type=int, default=0)
    ap.add_argument("--rate", type=float, help="positional rate; default = production pattern residual (PROD res_K)")
    ap.add_argument("--stats", default="/tmp/nestquant/19-capture")
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(12 / 80)
    torch.backends.cuda.matmul.allow_tf32 = False
    import nq19_load
    cap = nq19_load.Capture(root=a.stats)
    os.makedirs(RES, exist_ok=True)
    L = a.layer
    for E in map(int, a.experts.split(",")):
        path = f"{RES}/L{L}_E{E}.json"
        R = json.load(open(path)) if os.path.exists(path) else {"layer": L, "expert": E, "rate": a.rate, "eval": {}, "info": {}}
        t0 = time.time()
        dv = cap.expert_data(L, E, "val")
        dm = cap.expert_data(L, E, "matched")
        Ws = dv.teacher
        cache = {}
        for arm in (a.arms.split(";") if ";" in a.arms else a.arms.split(",")):
            spec, *opt = arm.split("/")
            kk = next((float(o[1:]) for o in opt if o.startswith("k")), None)
            cf = next((float(o[3:]) for o in opt if o.startswith("cap")), None)
            w = NB.parse_bnd(spec)
            tag = NB.bnd_tag(w) + (f"k{kk:g}" if kk else "") + (f"cap{cf:g}" if cf else "")
            if all(f"{m}@{tag}" in R["eval"].get("val", {}) for m in ("nq/L2", "nq/L4", "EXL3-2", "EXL3-4")) \
                    and R["info"].get(tag, {}).get("inner") == a.inner:
                continue
            HG = NB.glm_H_bnd(cap, L, E, w, k=kk, cap_frac=cf, cache=cache)
            methods = {}
            for K in (2, 4):
                q = []
                for pi, pn in enumerate(PROJ):
                    Wq, _ = h.quantize_exl3_like(Ws[pi], HG["H"][pi], K, count=1, sigma_reg=SIG[pn])
                    q.append(Wq.cpu()); h.free_scratch()
                methods[f"EXL3-{K}@{tag}"] = q
            art, dense = NE.encode_expert(Ws, HG, rate=a.rate, canonical_base=bool(a.canon), inner=a.inner)
            methods[f"nq/L2@{tag}"] = dense[2]; methods[f"nq/L4@{tag}"] = dense[4]
            R["info"][tag] = dict(meta={k: v for k, v in HG["meta"].items() if not torch.is_tensor(v)}, canon=a.canon, inner=a.inner,
                                  nq={p: art["meta"]["info"][p]["bits"] for p in PROJ})
            del art, HG
            for split, data in (("matched", dm), ("val", dv)):
                tb = tables(data, methods, bnd=(split == "val"), groups=(split == "matched"))
                R["eval"].setdefault(split, {}).update({k: v for k, v in tb.items() if k != "_rows"})
                R.setdefault("rows", {})[split] = tb["_rows"]
            json.dump(R, open(path + ".tmp", "w"), indent=1); os.replace(path + ".tmp", path)
            print(f"[L{L} E{E}] {tag} done {time.time()-t0:.0f}s", flush=True)
            del methods; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
