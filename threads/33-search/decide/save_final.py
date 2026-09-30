"""final (calib-selected) arm: V = 0.5*h16 + 0.5*h64/4, mult hysteresis; save heldout per-block scores + curve."""
import os, numpy as np
from multiprocessing import Pool
import dlib as D
OUTD = f"{D.W}/scores_glm52-heldout"; os.makedirs(OUTD, exist_ok=True)
HMS = (0.3, 0.35, 0.4, 0.45)
def job(L):
    c = "glm52-heldout"
    V = 0.5 * D.load_S("h16", c, L).astype(np.float64) + 0.5 * D.load_S("h64", c, L).astype(np.float64) / 4
    np.save(f"{OUTD}/L{L}.npy", V.astype(np.float32))
    bs, _ = D.load_blk(c, L); fx = D.fx_mask(L); fd = D.fd_mask(L)
    return [D.evaluate(D.replay(V, V * (1 + h), fx, fd, D.NBC, 0), bs, fx) for h in HMS]
with Pool(20) as p: R = np.array(p.map(job, D.LAYERS)).mean(0)
for h, (s, c) in zip(HMS, R): print(f"blend h16/h64 hm {h}: {s*100:.2f} / churn {c:.2f}")
