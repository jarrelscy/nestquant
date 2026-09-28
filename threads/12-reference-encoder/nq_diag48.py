"""dist48 failure diagnosis: nq vs EXL3 stage by stage on the same H (Capture.glm_H), per projection.

  CUDA_VISIBLE_DEVICES=2 python nq_diag48.py 30:169,3:2,3:60 [--eval]

Per projection, relative error in the original domain e_H = tr(E H E^T)/tr(W H W^T) (H = raw glm_H input Gram),
e_R = same with H_routed (routed rows only, the matched/routed eval metric proxy) and, for gate/up, the output-weighted
e_G = tr(G E H E^T)/tr(G W H W^T). Arms:
  EXL3-2 ldlq / noLDLQ / intra0 (LDLQ feedback only across 128-row blocks = nq's unit granularity)
  nq L2/L4: default (G two-sided, sign, lam .3, inner 0), inner2, noG (one-sided), novar, lam0
--eval: whole-expert matched eval (routed/forced) for the arms -> results_diag48/L{L}_E{E}.json
"""
import os, sys, json, time, argparse
os.environ.setdefault("OMP_NUM_THREADS", "16")
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/home/coder/git/nestquant/threads/19-full-capture")
import harness as h
import nq_encode as NE

PROJ = NE.PROJ
SIG = NE.PROD["sigma"]
PK = NE.PROD["res_K"]
OUT = f"{HERE}/results_diag48"


def exl3(W, H, K, sig, mode):
    Qm = h._ex()
    orig = Qm.block_ldl
    if mode == "intra0":
        def patched(Hm, b, *a, **kw):
            L, Hr = orig(Hm, b, *a, **kw)
            k = L.shape[0]
            for i in range(0, k, 128):
                L[i:i + 128, i:i + 128] = 0
            return L, Hr
        Qm.block_ldl = patched
    try:
        Wq, info = h.quantize_exl3_like(W, H, K, count=1, sigma_reg=sig, ldlq=(mode != "noldlq"))
    finally:
        Qm.block_ldl = orig
    h.free_scratch()
    return Wq.cpu(), info


def rel(W, Wq, H, G=None):
    W = W.cuda().float(); E = Wq.cuda().float() - W; H = H.cuda().float()
    if G is None:
        return float(((E @ H) * E).sum() / ((W @ H) * W).sum())
    G = G.cuda().float()
    return float(((G @ E @ H) * E).sum() / ((G @ W @ H) * W).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experts")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--arms", default="default,inner2,noG,novar,lam0")
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(12 / 80)
    torch.backends.cuda.matmul.allow_tf32 = False
    import nq19_load
    cap = nq19_load.Capture()
    os.makedirs(OUT, exist_ok=True)
    for spec in a.experts.split(","):
        L, E = map(int, spec.split(":"))
        t0 = time.time()
        dm = cap.expert_data(L, E, "matched")
        HG = cap.glm_H(L, E)
        Ws = dm.teacher
        comp = cap.components(L, E, keys=("A2", "D2"))
        res = {"layer": L, "expert": E, "proj": {}, "hg_meta": {k: v for k, v in HG.get("meta", {}).items()
                                                                 if not torch.is_tensor(v)}}
        # routed-only Grams (routed x x^T for gate/up input; routed a a^T for down input)
        HR = [comp.get("A2"), comp.get("A2"), comp.get("D2")]
        methods = {}
        for pi, pn in enumerate(PROJ):
            W = Ws[pi]; H = HG["H"][pi]; G = HG["G"][pi]
            r = {}
            def score(name, Wq):
                d = dict(eH=rel(W, Wq, H))
                if HR[pi] is not None:
                    d["eR"] = rel(W, Wq, HR[pi])
                if G is not None:
                    d["eG"] = rel(W, Wq, H, G)
                r[name] = d
            Hd = H.cuda().float().diagonal()
            ev = torch.linalg.eigvalsh(H.cuda().double())
            ev = ev.clamp_min(0)
            r["struct"] = dict(k=W.shape[1], n=W.shape[0],
                               eff_rank_H=float(ev.sum() ** 2 / (ev ** 2).sum()),
                               top1_share=float(ev[-1] / ev.sum()), top16_share=float(ev[-16:].sum() / ev.sum()),
                               diag_cv=float(Hd.std() / Hd.mean()),
                               row_norm_cv=float(W.float().norm(dim=1).std() / W.float().norm(dim=1).mean()),
                               col_norm_cv=float(W.float().norm(dim=0).std() / W.float().norm(dim=0).mean()),
                               w_absmax_over_rms=float(W.float().abs().max() / W.float().pow(2).mean().sqrt()))
            for K in (2, 4):
                for mode in ("ldlq", "noldlq", "intra0") if K == 2 else ("ldlq",):
                    Wq, info = exl3(W, H, K, SIG[pn], mode)
                    score(f"EXL3-{K}/{mode}", Wq)
                    r[f"EXL3-{K}/{mode}"]["proxy"] = float(info["proxy"])
                    if mode == "ldlq":
                        r["struct"].update({f"gs{K}": info["g_scale"], "aos": info["apply_out_scales"], "skew": info["skew"]})
                    if a.eval and mode in ("ldlq", "intra0"):
                        methods.setdefault(f"EXL3-{K}/{mode}", [None] * 3)[pi] = Wq
            res["proj"][pn] = r
            print(f"[{L}:{E}] {pn} exl3 done {time.time()-t0:.0f}s", flush=True)
        for arm in a.arms.split(","):
            kw = dict(res_K=PK, canonical_base=False, inner=0, lam=NE.PROD["lam"], base_var=NE.PROD["base_var"])
            HGa = HG
            if arm == "inner2":
                kw["inner"] = 2
            elif arm == "noG":
                HGa = dict(HG, G=[None, None, None])
            elif arm == "novar":
                kw["base_var"] = None
            elif arm == "lam0":
                kw["lam"] = 0.0
            elif arm.startswith("lr"):                   # lr[_tau<T>][_r<R>][_lam0]
                lc = dict(NE.LR)
                for t in arm.split("_")[1:]:
                    if t.startswith("tau"):
                        lc["tau"] = float(t[3:])
                    elif t.startswith("r"):
                        lc["rmax"] = int(t[1:])
                    elif t == "lam0":
                        kw["lam"] = 0.0
                kw["lr"] = lc
            elif arm.startswith("ocol"):                 # ocol[_tau<T>][_lam0]
                oc = dict(NE.OCOL)
                for t in arm.split("_")[1:]:
                    if t.startswith("tau"):
                        oc["tau"] = float(t[3:])
                    elif t == "lam0":
                        kw["lam"] = 0.0
                kw["ocol"] = oc
            t1 = time.time()
            art, dense = NE.encode_expert(Ws, HGa, **kw)
            dt = time.time() - t1
            for pi, pn in enumerate(PROJ):
                r = res["proj"][pn]
                for Lv in (2, 4):
                    W = Ws[pi]; Wq = dense[Lv][pi]
                    d = dict(eH=rel(W, Wq, HG["H"][pi]))
                    if HR[pi] is not None:
                        d["eR"] = rel(W, Wq, HR[pi])
                    if HG["G"][pi] is not None:
                        d["eG"] = rel(W, Wq, HG["H"][pi], HG["G"][pi])
                    d["proxy_rot"] = art["meta"]["info"][pn]["proxy_rot"][Lv]
                    r[f"nq_{arm}/L{Lv}"] = d
            if a.eval:
                for Lv in (2, 4):
                    methods[f"nq_{arm}/L{Lv}"] = dense[Lv]
            res.setdefault("nq_meta", {})[arm] = dict(rate=art["meta"]["rate"], ocol_idx=art["meta"].get("ocol_idx"),
                                                      lr_rank=art["meta"].get("lr_rank"),
                                                      bitexact={p: art["meta"]["info"][p]["bitexact"] for p in PROJ},
                                                      bits={p: art["meta"]["info"][p]["bits"] for p in PROJ}, time_s=dt)
            print(f"[{L}:{E}] nq {arm} {dt:.0f}s rate {art['meta']['rate']:.4f} ocol {art['meta'].get('ocol_idx')} lr {art['meta'].get('lr_rank')}", flush=True)
            del art, dense; torch.cuda.empty_cache()
        if a.eval:
            names = list(methods); ev = {}
            for i in range(0, len(names), 3):
                grp = {n: [w.cuda() for w in methods[n]] for n in names[i:i + 3]}
                tb = h.table(h.evaluate(dm, grp, groups=True))
                ev.update({n: tb[n] for n in grp})
                del grp; torch.cuda.empty_cache()
            res["eval"] = ev
        op = f"{OUT}/L{L}_E{E}.json"
        if os.path.exists(op):                        # merge with earlier arms
            old = json.load(open(op))
            for pn in PROJ:
                old["proj"][pn].update(res["proj"][pn])
            old.setdefault("eval", {}).update(res.get("eval", {}))
            old.setdefault("nq_meta", {}).update(res.get("nq_meta", {}))
            res = old
        json.dump(res, open(op, "w"), indent=1)
        # compact print
        for pn in PROJ:
            r = res["proj"][pn]
            print(pn, json.dumps(r["struct"]))
            for m, d in r.items():
                if m != "struct":
                    print(f"  {m:20s} " + " ".join(f"{k}={v:.5f}" for k, v in d.items()))
        if a.eval:
            for n, t in res["eval"].items():
                print(f"  EVAL {n:20s} routed {t.get('all/routed')} forced {t.get('all/forced')}")
        print(f"[{L}:{E}] total {time.time()-t0:.0f}s", flush=True)
        del dm, HG, comp; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
