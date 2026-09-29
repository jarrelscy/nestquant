"""T27: uniform (not routed) chunk-0 rows of a layer for the forced-metric regularizer.
  python nq27_frows.py L [N=40960]  -> /tmp/nestquant/27-pv-tune/frows/L{L}.pt  (x bf16 [N,6144], blk int32)
Rows drawn uniformly without replacement over s00-s05 (seed 2770+L); blk = global 2048-token block (same split rule).
"""
import os, sys, json
import numpy as np
import torch
from nq27_rows import ROOT, SHARDS, Dm

OUT = "/tmp/nestquant/27-pv-tune/frows"


def main():
    L = int(sys.argv[1]); N = int(sys.argv[2]) if len(sys.argv) > 2 else 40960
    os.makedirs(OUT, exist_ok=True)
    ns = [json.load(open(f"{ROOT}/{s}/acts/L{L}/done.json"))["rows"] for s in SHARDS]
    tot = sum(ns); rng = np.random.default_rng(2770 + L)
    g = np.sort(rng.choice(tot, N, replace=False))
    xs, base = [], 0
    for s, n in zip(SHARDS, ns):
        r = g[(g >= base) & (g < base + n)] - base
        x = np.memmap(f"{ROOT}/{s}/acts/L{L}/x.bf16", np.uint16, "r", shape=(n, Dm))
        xs.append(torch.from_numpy(np.ascontiguousarray(x[r])).view(torch.bfloat16)); base += n
    out = dict(layer=L, x=torch.cat(xs), blk=torch.from_numpy((g // 2048).astype(np.int32)))
    torch.save(out, f"{OUT}/L{L}.pt.tmp"); os.replace(f"{OUT}/L{L}.pt.tmp", f"{OUT}/L{L}.pt")
    print(L, len(g))


if __name__ == "__main__":
    main()
