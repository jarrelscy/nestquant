"""parity: v2 on glm52-heldout via plib (expert-indexed features, own replay) -> expect 74.77 / 2.78"""
import os, sys
os.environ["OMP_NUM_THREADS"] = "1"
from multiprocessing import Pool
import numpy as np
import plib as P
corpus = sys.argv[1] if len(sys.argv) > 1 else "glm52-heldout"

def job(L):
    import lightgbm as lgb
    b = lgb.Booster(model_file=P.V2)
    D = P.load(corpus, L)
    S = P.gbdt_scores(b, D["F"])
    return L, P.metric(P.replay(S, L, D["segs"]), L, D["bsal"], D["bcnt"], D["segs"])

with Pool(int(os.environ.get("NPROC", "16"))) as p:
    r = dict(p.map(job, P.T.LAYERS))
print(corpus, {k: round(float(np.mean([r[L][k] for L in r])), 4) for k in r[3]})
