"""reproduce v2 74.77 / 2.78 on glm52-heldout with glib's eval path (+ sm120 features path check on heldout rows)."""
import sys
from multiprocessing import Pool
import numpy as np
import glib as g

M = "/tmp/nestquant/32-gbdt-sal/models/v2_sal_tweedie1.5.txt"
corpus = sys.argv[1] if len(sys.argv) > 1 else "glm52-heldout"


def job(L):
    import lightgbm as lgb
    b = g.load_base(corpus, L)
    bst = lgb.Booster(model_file=M)
    nb = b["X9"].shape[0]
    S = bst.predict(b["X9"].reshape(-1, 9), num_threads=1).reshape(nb, 256).astype(np.float32)
    return L, g.metrics(S, L, b, hms=(0.5,))


if __name__ == "__main__":
    with Pool(20) as p:
        res = dict(p.map(job, g.T.LAYERS))
    print(g.summarize(res))
