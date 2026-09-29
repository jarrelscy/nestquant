"""T29 layer scan (CPU, H-only): predicted down-projection LDLQ cost of the 128-row ring layout under the block-Had128
input rotation vs a shard-local Had256 / full Had2048 rotation, relative to EXL3 (16-row LDLQ, Had128), metric =
undamped calibration H (proxy for val; diag29 showed val is ~1.1x harsher).  Damped sigma 1.0 as production.
  CUDA_VISIBLE_DEVICES= python scan29.py --layers 3,6,... --n 8   -> /tmp/nestquant/29-outlier-gap/scan.json"""
import os, sys, json, time, argparse
os.environ.setdefault("OMP_NUM_THREADS", "16")
import torch
import diag29 as D29

OUT = "/tmp/nestquant/29-outlier-gap/scan.json"


def had(n):
    Hm = torch.ones(1, 1, dtype=torch.float64)
    while Hm.shape[0] < n:
        Hm = torch.cat([torch.cat([Hm, Hm], 1), torch.cat([Hm, -Hm], 1)], 0)
    return Hm / n ** 0.5


def main():
    import nq_layer as NL
    ap = argparse.ArgumentParser(); ap.add_argument("--layers"); ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()
    hcap = NL.open_stats(f"{D29.R}/_stats", f"{D29.R}/_stats_mm", 0.25)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for L in map(int, a.layers.split(",")):
        if str(L) in res:
            continue
        t0 = time.time()
        g = torch.Generator().manual_seed(L)
        ex = torch.randperm(256, generator=g)[:a.n].tolist()
        rows = []
        for E in ex:
            H = hcap.glm_H(L, E, device="cpu")["H"][2].double()
            if not torch.isfinite(H).all() or float(H.diagonal().mean()) <= 0:
                continue
            k = H.shape[0]
            s = torch.randn(k, generator=torch.Generator().manual_seed(91426)).sign().double()
            M = H / H.diagonal().mean(); Hd = M + torch.eye(k, dtype=torch.float64)
            w = torch.linalg.eigvalsh(M).clamp_min(0)
            R128 = torch.block_diag(*[had(128)] * (k // 128)) * s[None]
            ref = D29.trD(R128 @ Hd @ R128.T, 16, [R128 @ M @ R128.T])[1]
            o = dict(E=E, eff_rank=float(w.sum() ** 2 / w.square().sum()))
            for nm, R in (("had128", R128), ("had256", torch.block_diag(*[had(k // 8)] * 8) * s[None]), ("had_full", had(k) * s[None])):
                o[nm] = D29.trD(R @ Hd @ R.T, 128, [R @ M @ R.T])[1] / ref
            rows.append(o)
        res[str(L)] = rows
        mean = lambda kk: sum(r[kk] for r in rows) / len(rows)
        print(L, len(rows), " ".join(f"{kk} {mean(kk):.2f}" for kk in ("eff_rank", "had128", "had256", "had_full")),
              f"{time.time()-t0:.0f}s", flush=True)
        json.dump(res, open(OUT, "w"))


if __name__ == "__main__":
    main()
