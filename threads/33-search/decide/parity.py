import sys, numpy as np
from multiprocessing import Pool
import dlib as D
corpus = sys.argv[1]
HMS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
def job(L):
    S = D.load_S("v2", corpus, L).astype(np.float64); bs, bc = D.load_blk(corpus, L); fx = D.fx_mask(L); fd = D.fd_mask(L)
    out = []
    for hm in HMS:
        sv = D.replay(S, S * (1 + hm), fx, fd, D.NBC, 0)
        out.append(D.evaluate(sv, bs, fx))
    return out
with Pool(16) as p: R = np.array(p.map(job, D.LAYERS))
for i, hm in enumerate(HMS):
    print(f"hm {hm:.2f}  sal {R[:, i, 0].mean()*100:.2f}  churn {R[:, i, 1].mean():.3f}")
