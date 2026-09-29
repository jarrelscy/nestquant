import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scalelib as S
V2 = "/tmp/nestquant/32-gbdt-sal/models/v2_sal_tweedie1.5.txt"
def job(L):
    import lightgbm as lgb
    corpus = "glm52-heldout"
    D = S.load(corpus, L); F, e256 = S.feats(D)
    d = np.load(f"{S.SRC}/rows_bandall/{corpus}/L{L}.npz"); d2 = np.load(f"{S.SRC}/rows_v2_bandall/{corpus}/L{L}.npz")
    cand = d["cand"].astype(np.int64)
    ref = np.concatenate([d["X"], d2["X2"]], -1)
    F2 = F.copy(); np.put_along_axis(F2, cand[..., None], ref, 1)
    mine = np.take_along_axis(F, cand[..., None], 1)
    rel = (np.abs(mine - ref) / (np.abs(ref) + 1e-6)).max((0, 1))
    b = lgb.Booster(model_file=V2)
    out = [rel]
    for FF in (F, F2):
        sv = S.replay(S.predict_S(b, FF, L), L, D["sg"]); out.append(S.block_metrics(sv, D, L))
    return L, out
with Pool(16) as p: R = dict(p.map(job, S.LAYERS))
np.set_printoptions(linewidth=200); print("max rel", np.max([R[L][0] for L in R], 0))
print("mine", S.summarize({L: R[L][1] for L in R})); print("ref-feats", S.summarize({L: R[L][2] for L in R}))
