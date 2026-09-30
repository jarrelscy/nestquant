"""CPU backup eval: score NAME on a chain subset of sm120tf (chains i % MOD == 0) and replay v2 vs NAME on the same
chains (k0, sync lag 0; churn within chains only, no reset transitions - same convention for both arms).
  LAYOUT=k0 python tf_subset.py NAME MOD hms"""
import os
import sys
from multiprocessing import Pool

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train as TR                                  # noqa: E402  (before jlib: it prepends 32-gbdt-sal/)
import jlib as J                                    # noqa: E402

name, MOD = sys.argv[1], int(sys.argv[2])
hms = [float(x) for x in sys.argv[3].split(",")]
TD = f"{J.OUT}/tmp_tfX"


def job(i):
    torch.set_num_threads(1)
    L = TR.LAYERS[i]
    d = J.load("sm120tf", L)
    sel = [c for c in range(len(d["sg"])) if c % MOD == 0]
    idx = np.concatenate([np.arange(*d["sg"][c]) for c in sel])
    z = np.load(f"{TD}/L{L}.npz")
    X = torch.from_numpy(z["X"][idx]); P = z["P"][idx]
    ck = torch.load(f"{J.OUT}/models/{name}.pt", map_location="cpu"); a = ck["args"]
    net = TR.Net(ck["K"], a["arch"], a["d"], a["nl"]).eval(); net.load_state_dict(ck["state"])
    lp = torch.from_numpy(np.log(np.maximum(P, 1e-30)).astype(np.float32))
    fx = torch.from_numpy(J.masks(L)[0])[None].expand(len(idx), -1)
    li = torch.full((len(idx),), i, dtype=torch.long)
    out = []
    with torch.no_grad():
        for j in range(0, len(idx), 512):
            out.append(torch.exp((lp[j:j + 512] + net(X[j:j + 512], lp[j:j + 512], li[j:j + 512], fx[j:j + 512])).clamp(max=30)))
    Sj = np.zeros((d["bcnt"].shape[0], J.NE), np.float32); Sj[idx] = torch.cat(out).numpy()
    Sv = np.zeros_like(Sj); Sv[idx] = P
    return {(arm, hm): J.metric(S, d, L, hm, 0.0, sel) for arm, S in (("v2", Sv), (name, Sj)) for hm in hms}


if __name__ == "__main__":
    with Pool(int(os.environ.get("NPROC", "10"))) as p:
        R = p.map(job, range(len(TR.LAYERS)))
    for arm in ("v2", name):
        pts = []
        for hm in hms:
            n, dn, cs, cn = [sum(r[(arm, hm)][q] for r in R) for q in range(4)]
            pts.append((cs / cn, n / dn)); print(f"{arm:14s} hm {hm:.2f} sal-hot {100 * n / dn:.2f} churn {cs / cn:.2f}")
        x, y = zip(*sorted(pts))
        for c in (2.78, 3.2):
            print(f"{arm:14s} @churn {c}: {100 * np.interp(c, x, y) if x[0] <= c <= x[-1] else float('nan'):.2f}")
