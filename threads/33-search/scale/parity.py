"""parity: my features vs T32 rows (band all), and v2 heldout sal-hot/churn via my eval path."""
import os, sys, json
os.environ.setdefault("OMP_NUM_THREADS", "1")
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scalelib as S
V2 = "/tmp/nestquant/32-gbdt-sal/models/v2_sal_tweedie1.5.txt"
corpus = sys.argv[1] if len(sys.argv) > 1 else "glm52-heldout"

def job(L):
    import lightgbm as lgb
    D = S.load(corpus, L)
    F, e256 = S.feats(D)
    d = np.load(f"{S.SRC}/rows_bandall/{corpus}/L{L}.npz"); d2 = np.load(f"{S.SRC}/rows_v2_bandall/{corpus}/L{L}.npz")
    cand = d["cand"].astype(np.int64)
    ref = np.concatenate([d["X"], d2["X2"]], -1)
    mine = np.take_along_axis(F, cand[..., None], 1)
    err = np.abs(mine - ref).max((0, 1)) / (np.abs(ref).max((0, 1)) + 1e-9)
    b = lgb.Booster(model_file=V2)
    Sm = S.predict_S(b, F, L)
    sv = S.replay(Sm, L, D["sg"])
    M = S.block_metrics(sv, D, L)
    return L, err.tolist(), M

if __name__ == "__main__":
    with Pool(16) as p:
        R = p.map(job, S.LAYERS)
    err = np.max([r[1] for r in R], 0)
    print("max rel feat err", dict(zip(S.V2F, np.round(err, 6))))
    print("v2", S.summarize({L: m for L, _, m in R}))
