"""T29: pick high-KLD-band experts by val top-10 routed output-energy share (same definition as T27 nq27_ess 'val').
  python sel29.py L1,L2,... [--min-rows 256] [--cand 4]  -> /tmp/nestquant/29-outlier-gap/sel_band.json
Stage 1 (cheap): top-10 share of p^2 ||x||^2 per expert; stage 2: exact p^2 ||y_T(x)||^2 for the top --cand per layer."""
import os, sys, json, argparse
import torch
import harness as h
from orbit_duet.source import weights
SRC = "/tmp/nestquant/src/glm53-fp8"; R = "/tmp/nestquant/nq-encode-v1"; OUT = "/tmp/nestquant/29-outlier-gap"


def share(w):
    s = w.sort(descending=True).values
    return float(s[:10].sum() / s.sum()), float(s.sum() ** 2 / s.square().sum())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("layers"); ap.add_argument("--min-rows", type=int, default=256)
    ap.add_argument("--cand", type=int, default=4); a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(12 / 80)
    res = {}
    for L in map(int, a.layers.split(",")):
        cap = torch.load(f"{R}/_stats/eval/val/layer_{L}.pt", weights_only=True, mmap=True)
        ids, p = cap["ids"], cap["p"]
        xn = torch.cat([cap["x"][i:i + 8192].cuda().float().square().sum(-1).cpu() for i in range(0, len(ids), 8192)])
        st = {}
        for E in range(256):
            r, s = torch.where(ids == E)
            if len(r) < a.min_rows:
                continue
            st[E] = (share(xn[r].double() * p[r, s].double().square())[0], r, s)
        top = sorted(st, key=lambda e: -st[e][0])[:a.cand]
        out = {}
        for E in top:
            _, r, s = st[E]
            tb = [w.to("cuda", torch.bfloat16) for w in weights(SRC, L, E, device="cpu")]
            y = torch.cat([h._teacher(cap["x"][r[i:i + 2048]].cuda(), tb).double().square().sum(-1).cpu()
                           for i in range(0, len(r), 2048)])
            t10, ess = share(y * p[r, s].double().square())
            out[E] = dict(rows=len(r), in_top10=st[E][0], top10=t10, ess=ess)
            print(L, E, out[E], flush=True)
        res[L] = out
        del cap
    json.dump(res, open(f"{OUT}/sel_band.json", "w"), indent=1)


if __name__ == "__main__":
    main()
