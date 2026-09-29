"""training / val rows from calib-fit.  rows.py NAME BAND SUB
  BAND  band = EMA256 ranks 20..120 of non-fixed (v2's rows) | all = all non-fixed
  train chains 0..25 (OOF priors P_f{chain%4}), val chains 26..31 (P_all); valid-horizon blocks, every SUB-th block.
-> $W/rows/NAME/{tr,va}_{X,y,aux}.npy   X cols = ALLX + v2 (v2 model's prediction), y = next-64 sal / m_L (train),
   aux = [layer, chain, block, rank_v2 among non-fixed, rank_e256]"""
import os
import sys
from multiprocessing import Pool
import numpy as np
import glib as g

name, band, sub = sys.argv[1], sys.argv[2], int(sys.argv[3])
OUT = f"{g.W}/rows/{name}"
V2 = "/tmp/nestquant/32-gbdt-sal/models/v2_sal_tweedie1.5.txt"
PZ = dict(np.load(f"{g.W}/priors.npz"))


def job(L):
    import lightgbm as lgb
    li = g.T.LAYERS.index(L)
    v2 = lgb.Booster(model_file=V2)
    res = {"tr": [], "va": []}
    nfx = np.ones(g.NE, bool); nfx[g.FIXED[L]] = False
    for f in ["f0", "f1", "f2", "f3", "all"]:
        b = g.full_feats("calib-fit", L, PZ[f"P_{f}"][li])
        XA = b["XA"]; nb = XA.shape[0]
        p2 = v2.predict(XA[..., :9].reshape(-1, 9), num_threads=1).reshape(nb, g.NE).astype(np.float32)
        y = b["ysal"] / b["mL"]
        sc = np.where(nfx[None], p2, -np.inf); rv = np.argsort(np.argsort(-sc, 1, kind="stable"), 1)
        se = np.where(nfx[None], XA[..., g.ALLX.index("e256")], -np.inf); re = np.argsort(np.argsort(-se, 1, kind="stable"), 1)
        for ci, (s, e) in enumerate(b["segs"]):
            if f == "all" and ci < g.NTRAIN_CH or f != "all" and (ci >= g.NTRAIN_CH or ci % 4 != int(f[1])):
                continue
            ks = np.arange(s, e); ks = ks[np.isfinite(y[ks, 0]) & ((ks - s) % sub == 0)]
            m = np.zeros((len(ks), g.NE), bool)
            m[:] = nfx[None]
            if band == "band":
                m &= (re[ks] >= 20) & (re[ks] <= 120)
            bi, ei = np.nonzero(m); kk = ks[bi]
            X = np.concatenate([XA[kk, ei], p2[kk, ei, None]], 1)
            aux = np.stack([np.full(len(kk), L), np.full(len(kk), ci), kk, rv[kk, ei], re[kk, ei]], 1).astype(np.int32)
            res["tr" if f != "all" else "va"].append((X, y[kk, ei].astype(np.float32), aux, ei))
    out = {}
    for k, v in res.items():
        out[k] = [np.concatenate([x[i] for x in v]) for i in range(3)]
    return L, out, float(b["mL"])


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    acc = {"tr": [[], [], []], "va": [[], [], []]}
    mL = {}
    with Pool(int(os.environ.get("NPROC", "16"))) as p:
        for L, out, m in p.imap(job, g.T.LAYERS):
            mL[L] = m
            for k in out:
                for i in range(3):
                    acc[k][i].append(out[k][i])
            print(L, len(out["tr"][1]), flush=True)
    for k in acc:
        for i, n in enumerate(("X", "y", "aux")):
            np.save(f"{OUT}/{k}_{n}.npy", np.concatenate(acc[k][i]))
            acc[k][i] = None
    np.save(f"{OUT}/mL.npy", np.array([mL[L] for L in g.T.LAYERS]))
    print("done")
