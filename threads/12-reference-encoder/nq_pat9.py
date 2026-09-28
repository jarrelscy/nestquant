"""9-expert validation of the production pattern residual (T14 down-heavy) + fast-encode settings.

  CUDA_VISIBLE_DEVICES=4 python nq_pat9.py 16:36,49:36 [--cfgs p4085_c1i2,p4085_c0i0,p4126_c1i2,p4126_c0i0]

cfg = p{4085|4126}_c{canonical 2-pass 1|single joint pass 0}i{inner}. Same H (nq_run.glm_H) and eval set as results_v1;
rows go into results_v1/L{L}_E{E}.json as nq_{cfg}/L2, nq_{cfg}/L4. Artifacts: /tmp/nestquant/12-reference-encoder/pat9.
"""
import os, sys, time, json, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch
import nq_run as R
import nq_encode as NE
import harness as h

PAT = {"p4085": {"gate": 1.9375, "up": 1.9375, "down": 2.3125}, "p4126": {"gate": 2.0, "up": 2.0, "down": 2.3125}}
ART = "/tmp/nestquant/12-reference-encoder/pat9"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experts")
    ap.add_argument("--cfgs", default="p4085_c1i2,p4085_c0i0,p4126_c1i2,p4126_c0i0")
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(12 / 80)
    torch.backends.cuda.matmul.allow_tf32 = False
    os.makedirs(ART, exist_ok=True)
    for spec in a.experts.split(","):
        L, E = map(int, spec.split(":"))
        book = R.Book(f"{R.RES}/L{L}_E{E}.json")
        ev = book.R.setdefault("eval", {})
        data = h.load_expert(L, E); HG = R.glm_H(data, L, E)
        methods, extra = {}, {}
        for cfg in a.cfgs.split(","):
            if f"nq_{cfg}/L4" in ev:
                continue
            pk, ci = cfg.split("_")
            lr = dict(NE.LR) if ci.endswith("lr") else None
            ci = ci[:-2] if lr else ci
            canon, inner = ci[1] == "1", int(ci[3:])
            torch.cuda.synchronize(); t0 = time.time()
            art, dense = NE.encode_expert(data.teacher, HG, res_K=PAT[pk], canonical_base=canon, inner=inner, lr=lr)
            torch.cuda.synchronize(); dt = time.time() - t0
            torch.save(art, f"{ART}/L{L}_E{E}_{cfg}.pt")
            m = art["meta"]
            for Lv in (2, 4):
                methods[f"nq_{cfg}/L{Lv}"] = dense[Lv]
                extra[f"nq_{cfg}/L{Lv}"] = dict(bpw=sum(m["info"][p]["bits"][Lv] for p in NE.PROJ) / 3, time_s=dt,
                                                res_K=PAT[pk], canonical=canon, inner=inner,
                                                L2_equal_canonical=[m["info"][p].get("L2_equal_canonical") for p in NE.PROJ],
                                                bitexact=[m["info"][p]["bitexact"] for p in NE.PROJ])
            print(f"[{L}:{E}] {cfg} L4 {m['rate']:.4f} {dt:.0f}s bitexact {[m['info'][p]['bitexact'] for p in NE.PROJ]}", flush=True)
            del art, dense; torch.cuda.empty_cache()
        if methods:
            R.evaluate_into(book, data, methods, extra)
        del data, HG; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
