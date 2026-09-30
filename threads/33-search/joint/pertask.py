"""sm120tf pooled + per-task sal-hot / churn for several score dirs (k0 via LAYOUT).  Chains of sm120tf = tasks.
  LAYOUT=k0 python pertask.py hms DIR1 [DIR2 ...]"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

import jlib as J

hms = [float(x) for x in sys.argv[1].split(",")]
DIRS = sys.argv[2:]
_m = json.load(open(f"{J.BLK}/sm120tf/meta.json"))
TASKS = [n for n, a, b in zip(_m["chains"], _m["bstart"][:-1], _m["bstart"][1:]) if b > a]   # = J.load sg order


def split_metric(S, d, L, hm):
    """one replay, per-task (num, den, churn sum, churn count) + 'pooled' (incl. chain-start transitions)."""
    fx, fd = J.masks(L)
    sv = J.replay(S, fx, fd, d["sg"], hm, 0.0)
    bs = d["bsal"].astype(np.float64)
    out = {}
    for t, (s, e) in enumerate(d["sg"]):
        h = sv[s:e] | fx
        out[t] = (float((bs[s:e] * h).sum()), float(bs[s:e].sum()), float((sv[s + 1:e] & ~sv[s:e - 1]).sum()), e - s - 1)
    n, dn, cs, cn = [sum(out[t][q] for t in out) for q in range(4)]
    for t, (s, e) in enumerate(d["sg"]):
        if s > 0:
            cs += float((sv[s] & ~sv[s - 1]).sum()); cn += 1
    out["pooled"] = (n, dn, cs, cn)
    return out


def job(L):
    d = J.load("sm120tf", L)
    out = {}
    for di, sd in enumerate(DIRS):
        S = np.load(f"{sd}/L{L}.npy").astype(np.float32)
        for hm in hms:
            for k, v in split_metric(S, d, L, hm).items():
                out[(di, hm, k)] = v
    return out


TSEL = [TASKS.index(t) for t in os.environ.get("TSEL", "").split(",") if t]   # pooled over a task subset (folds)


def curve(R, di, key, agg="lmean"):
    """agg 'lmean' = sweep.py (layer mean of sal-hot ratios); 'pooled' = salience summed over layers, then the share
    (tracks KLD per T32).  churn = layer mean in both (equal block counts per layer)."""
    pts = []
    for hm in hms:
        ks = TSEL if key == "sel" else [key]
        per = [[sum(r[(di, hm, k)][q] for k in ks) for q in range(4)] for r in R]
        sal = (np.mean([p[0] / p[1] for p in per]) if agg == "lmean" else
               sum(p[0] for p in per) / sum(p[1] for p in per))
        pts.append((float(np.mean([p[2] / p[3] for p in per])), 100 * float(sal)))
    return pts


def at(pts, c):
    x, y = zip(*sorted(pts))
    return np.interp(c, x, y) if x[0] <= c <= x[-1] else float("nan")


if __name__ == "__main__":
    with Pool(int(os.environ.get("NPROC", "16"))) as p:
        R = p.map(job, J.LAYERS)
    import pickle
    pk = os.environ.get("PKL")
    if pk:
        pickle.dump(dict(R=R, dirs=DIRS, hms=hms, tasks=TASKS), open(pk, "wb"))
    keys = (["sel"] + TSEL) if TSEL else (["pooled"] + list(range(len(TASKS))))
    for key in keys:
        nm = {"pooled": "POOLED", "sel": "POOLED(" + "+".join(TASKS[t][:5] for t in TSEL) + ")"}.get(key, None) or TASKS[key]
        for di, sd in enumerate(DIRS):
          for agg in ("lmean", "pooled"):
            pts = curve(R, di, key, agg)
            print(f"{nm:26s} {agg:6s} {os.path.basename(sd):28s} " + " ".join(f"{c:.2f}/{s:.2f}" for c, s in pts) +
                  f" | @2.74 {at(pts, 2.74):.2f} @2.78 {at(pts, 2.78):.2f} @3.2 {at(pts, 3.2):.2f}")
