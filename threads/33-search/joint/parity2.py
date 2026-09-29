import os, sys
from multiprocessing import Pool
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, jlib as J, lightgbm as lgb


def job(L):
    d = J.load("glm52-heldout", L)
    r = np.load(f"{J.T32}/rows_bandall/glm52-heldout/L{L}.npz"); x2 = np.load(f"{J.T32}/rows_v2_bandall/glm52-heldout/L{L}.npz")["X2"]
    X = np.concatenate([r["X"], x2], -1)
    b = lgb.Booster(model_file=J.V2)
    p = b.predict(X.reshape(-1, 9), num_threads=1).reshape(X.shape[:2])
    S = np.zeros((X.shape[0], 256), np.float32); np.put_along_axis(S, r["cand"].astype(int), p.astype(np.float32), 1)
    F = J.features(d); fx, _ = J.masks(L); P = J.v2_pred(F, L, fx)
    return J.metric(S, d, L), J.metric(P, d, L), float(np.abs(S[:, ~fx] - P[:, ~fx]).max())


if __name__ == "__main__":
    with Pool(16) as p:
        res = p.map(job, J.LAYERS)
    for k in (0, 1):
        r = np.array([x[k] for x in res]); print("sal-hot %.3f churn %.3f" % (100 * np.mean(r[:, 0] / r[:, 1]), np.mean(r[:, 2] / r[:, 3])))
    print(max(x[2] for x in res))
