"""L2 per-expert lambda-fallback A/B (lead 2026-09-28; production unchanged): p4126 single-pass inner0 at lam 0.3 vs 0.

  CUDA_VISIBLE_DEVICES=4 python nq_lam9.py 16:36,16:92,... [--seeds 91426,7]

Per expert and seed: nq lam0.3 and lam0 (L2, L4) + EXL3-2/EXL3-4 with the same H (nq_run.glm_H) and seed. Scored on
(1) the results_v1 orbit eval (all/control/ood, forced/routed) and (2) T19's pooled held-out val rows for the layer
(cap.expert_data(L, E, "val"), ~2k routed rows / expert) -> results_lam/L{L}_E{E}.json.  The seed-91426 lam0.3 run is
decoded from the nq_pat9 artifact (identical to re-encoding)."""
import os, sys, json, time, argparse
os.environ.setdefault("OMP_NUM_THREADS", "16")
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/home/coder/git/nestquant/threads/19-full-capture")
import nq_run as R
import nq_encode as NE
import nq_decode as D
import harness as h

PK = {"gate": 2.0, "up": 2.0, "down": 2.3125}
ART = "/tmp/nestquant/12-reference-encoder/pat9"
OUT = f"{HERE}/results_lam"


def ev(data, methods, groups):
    out = {}
    names = list(methods)
    for i in range(0, len(names), 4):
        grp = {n: [w.cuda() for w in methods[n]] for n in names[i:i + 4]}
        tb = h.table(h.evaluate(data, grp, groups=groups))
        out.update({n: tb[n] for n in grp})
        del grp; torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experts")
    ap.add_argument("--seeds", default="91426,7")
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(12 / 80)
    torch.backends.cuda.matmul.allow_tf32 = False
    import nq19_load
    cap = nq19_load.Capture(root="/tmp/nestquant/19-capture")
    os.makedirs(OUT, exist_ok=True)
    for spec in a.experts.split(","):
        L, E = map(int, spec.split(":"))
        path = f"{OUT}/L{L}_E{E}.json"
        res = json.load(open(path)) if os.path.exists(path) else {"layer": L, "expert": E, "v1": {}, "val": {}, "time": {}}
        data = h.load_expert(L, E); HG = R.glm_H(data, L, E)
        dv = cap.expert_data(L, E, "val")
        for seed in map(int, a.seeds.split(",")):
            if f"nq_lam0/L4@s{seed}" in res["val"]:
                continue
            methods = {}
            for K in (2, 4):
                q = []
                for pi, pn in enumerate(NE.PROJ):
                    Wq, _ = h.quantize_exl3_like(data.teacher[pi], HG["H"][pi], K, count=1, seed=seed,
                                                 sigma_reg=NE.PROD["sigma"][pn])
                    q.append(Wq.cpu()); h.free_scratch()
                methods[f"EXL3-{K}@s{seed}"] = q
            for lam in (0.3, 0.0):
                tag = f"nq_lam{lam:g}"
                apath = f"{ART}/L{L}_E{E}_p4126_c0i0.pt"
                if lam == 0.3 and seed == 91426 and os.path.exists(apath):
                    art = torch.load(apath, weights_only=False)
                    dense = {Lv: [w.cpu() for w in D.decode_expert(art, Lv)] for Lv in (2, 4)}
                else:
                    torch.cuda.synchronize(); t0 = time.time()
                    art, dense = NE.encode_expert(data.teacher, HG, res_K=PK, canonical_base=False, inner=0, lam=lam, seed=seed)
                    res["time"][f"{tag}@s{seed}"] = time.time() - t0
                for Lv in (2, 4):
                    methods[f"{tag}/L{Lv}@s{seed}"] = dense[Lv]
                del art, dense; torch.cuda.empty_cache()
            res["v1"].update(ev(data, methods, True))
            res["val"].update(ev(dv, methods, False))
            json.dump(res, open(path + ".tmp", "w"), indent=1); os.replace(path + ".tmp", path)
            print(f"[{L}:{E}] seed {seed} done", flush=True)
            del methods; torch.cuda.empty_cache()
        del data, HG, dv; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
