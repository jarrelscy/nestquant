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


def job(L):
    d = J.load("sm120tf", L)
    out = {}
    for di, sd in enumerate(DIRS):
        S = np.load(f"{sd}/L{L}.npy").astype(np.float32)
        for hm in hms:
            out[(di, hm, "pooled")] = J.metric(S, d, L, hm, 0.0, None)
            for t in range(len(d["sg"])):
                out[(di, hm, t)] = J.metric(S, d, L, hm, 0.0, [t])
    return out


TSEL = [TASKS.index(t) for t in os.environ.get("TSEL", "").split(",") if t]   # pooled over a task subset (folds)


def curve(R, di, key):
    pts = []
    for hm in hms:
        ks = TSEL if key == "sel" else [key]
        n, dn, cs, cn = [sum(r[(di, hm, k)][q] for r in R for k in ks) for q in range(4)]
        pts.append((cs / cn, 100 * n / dn))
    return pts


def at(pts, c):
    x, y = zip(*sorted(pts))
    return np.interp(c, x, y) if x[0] <= c <= x[-1] else float("nan")


if __name__ == "__main__":
    with Pool(int(os.environ.get("NPROC", "16"))) as p:
        R = p.map(job, J.LAYERS)
    keys = (["sel"] + TSEL) if TSEL else (["pooled"] + list(range(len(TASKS))))
    for key in keys:
        nm = {"pooled": "POOLED", "sel": "POOLED(" + "+".join(TASKS[t][:5] for t in TSEL) + ")"}.get(key, None) or TASKS[key]
        for di, sd in enumerate(DIRS):
            pts = curve(R, di, key)
            print(f"{nm:26s} {os.path.basename(sd):28s} " + " ".join(f"{c:.2f}/{s:.2f}" for c, s in pts) +
                  f" | @2.74 {at(pts, 2.74):.2f} @2.78 {at(pts, 2.78):.2f} @3.2 {at(pts, 3.2):.2f}")
