"""per-expert static priors from calib-fit train chains (0..25): fold 'all' (for val/heldout/sm120) + 4 OOF folds
(for training rows) -> $W/priors.npz  P_all [75,256,k], P_f0..P_f3"""
from multiprocessing import Pool
import numpy as np
import glib as g


def job(L):
    b = g.load_base("calib-fit", L)
    sg = b["segs"][:g.NTRAIN_CH]
    out = {"all": g.priors_from(b["bcnt"], b["bsal"], sg, b["ysal"], L)}
    for f in range(4):
        out[f"f{f}"] = g.priors_from(b["bcnt"], b["bsal"], [s for i, s in enumerate(sg) if i % 4 != f], b["ysal"], L)
    return L, out


if __name__ == "__main__":
    with Pool(20) as p:
        r = dict(p.map(job, g.T.LAYERS))
    np.savez(f"{g.W}/priors.npz", **{f"P_{k}": np.stack([r[L][k] for L in g.T.LAYERS]) for k in r[3]}, names=g.PRI)
    P = np.stack([r[L]["all"] for L in g.T.LAYERS])
    for i, n in enumerate(g.PRI):
        print(n, np.percentile(P[..., i], [5, 50, 95]).round(3))
