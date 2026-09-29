"""T29 CPU prediction: down-projection val cost of nq (128-row rings) with a down low-rank plane of rank r and damping
sigma, relative to EXL3 (16-row LDLQ, no lr, production sigma 1.0).  White-noise LDLQ model of diag29.
  CUDA_VISIBLE_DEVICES= python diag29_rank.py L:E ...   -> /tmp/nestquant/29-outlier-gap/diag_r/L{L}_E{E}.json"""
import os, sys, json, time
os.environ.setdefault("OMP_NUM_THREADS", "32")
import torch
import diag29 as D29

OUT = "/tmp/nestquant/29-outlier-gap/diag_r"


def main():
    import nq_layer as NL, nq_encode as NE
    hcap = NL.open_stats(f"{D29.R}/_stats", f"{D29.R}/_stats_mm", 0.25)
    os.makedirs(OUT, exist_ok=True)
    for pr in sys.argv[1:]:
        L, E = map(int, pr.split(":"))
        o = f"{OUT}/L{L}_E{E}.json"
        if os.path.exists(o):
            continue
        t0 = time.time()
        H = hcap.glm_H(L, E, device="cpu")["H"][2].double()
        Hv = D29.val_grams(L, E)[2]
        k = H.shape[0]
        signs = torch.randn(k, generator=torch.Generator().manual_seed(91426)).sign().double()
        ev = [("val", Hv)]
        ref = D29.analyse(H, 1.0, signs=signs, evals=ev)["c16_val"]
        res = dict(layer=L, expert=E, ref_c16_val=ref, arms={})
        for sig in (1.0, 0.25, 0.05):
            for r in (0, 2, 4, 6, 8, 12, 16):
                V = NE.lr_detect(H.float(), tau=0.0, rmax=r).float() if r else None
                a = D29.analyse(H, sig, V=V, signs=signs, evals=ev)
                res["arms"][f"s{sig}_r{r}"] = dict(nq=a["c128_val"] / ref, exl3=a["c16_val"] / ref)
        json.dump(res, open(o, "w"), indent=1)
        print(L, E, {kk: round(v["nq"], 2) for kk, v in res["arms"].items()}, f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
