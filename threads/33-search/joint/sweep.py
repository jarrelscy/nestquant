"""sweep.py STREAM SCOREDIR [hm,...] [--sel val] -> sal-hot/churn per hm for per-layer score files SCOREDIR/L{L}.npy
(positive scores [nb,256]).  Writes SCOREDIR/sweep_STREAM[_val].json.  --sel val = calib-fit chains 28..31 only."""
import json
import os
import sys
from multiprocessing import Pool
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import jlib as J

VAL = list(range(28, 32))


def job(a):
    stream, sd, L, hms, sel = a
    d = J.load(stream, L)
    S = np.load(f"{sd}/L{L}.npy").astype(np.float32)
    if S.shape[0] != d["bcnt"].shape[0]:
        raise ValueError((L, S.shape))
    return L, [J.metric(S, d, L, hm=h, sel=sel) for h in hms]


def run(stream, sd, hms, sel=None, nproc=16):
    with Pool(nproc) as p:
        res = dict(p.map(job, [(stream, sd, L, hms, sel) for L in J.LAYERS]))
    out = {}
    for i, h in enumerate(hms):
        r = np.array([res[L][i] for L in J.LAYERS])
        out[str(h)] = dict(sal=float(np.mean(r[:, 0] / r[:, 1])), churn=float(np.mean(r[:, 2] / r[:, 3])),
                           per_layer=(r[:, 0] / r[:, 1]).tolist())
        print(f"  hm {h:5.2f}  sal-hot {out[str(h)]['sal'] * 100:6.2f}  churn {out[str(h)]['churn']:5.2f}", flush=True)
    return out


def interp(out, c):
    """sal-hot at churn c by linear interpolation over the hm curve."""
    pts = sorted((v["churn"], v["sal"]) for v in out.values())
    x = [p[0] for p in pts]; y = [p[1] for p in pts]
    if c < x[0] or c > x[-1]:
        return float("nan")
    return float(np.interp(c, x, y))


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    sel = VAL if "--val" in sys.argv else None
    hms = [float(x) for x in a[2].split(",")] if len(a) > 2 else [0.0, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5]
    out = run(a[0], a[1], hms, sel, int(os.environ.get("NPROC", "16")))
    tag = a[0] + ("_val" if sel else "")
    for c in (2.78, 3.2):
        print(f"  @churn {c}: sal-hot {interp(out, c) * 100:.2f}")
    f = f"{a[1]}/sweep_{tag}.json"
    old = json.load(open(f)) if os.path.exists(f) else {}
    old.update(out)
    json.dump(old, open(f, "w"), indent=1)
