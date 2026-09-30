"""CPU: score other v2-recipe boosters (all 256 experts) on a stream with the exact jlib v2 features.
  boostjob.py STREAM name=model.txt[,name2=...] -> scores/{name}_{STREAM}/L.npy f32 [nb,256]"""
import os, sys
from multiprocessing import Pool
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import jlib as J
stream = sys.argv[1]
B = dict(kv.split("=", 1) for kv in sys.argv[2].split(","))


def job(L):
    todo = {n: p for n, p in B.items() if not os.path.exists(f"{J.OUT}/scores/{n}_{stream}/L{L}.npy")}
    if not todo:
        return L
    import lightgbm as lgb
    F = J.features(J.load(stream, L))
    for n, p in todo.items():
        b = lgb.Booster(model_file=p)
        X = np.stack([F[f][:, :] for f in b.feature_name()], -1).reshape(-1, len(b.feature_name()))
        P = b.predict(X, num_threads=1).reshape(-1, J.NE).astype(np.float32)
        f = f"{J.OUT}/scores/{n}_{stream}/L{L}.npy"
        np.save(f + ".part.npy", P); os.replace(f + ".part.npy", f)
    return L


if __name__ == "__main__":
    for n in B:
        os.makedirs(f"{J.OUT}/scores/{n}_{stream}", exist_ok=True)
    with Pool(int(os.environ.get("NPROC", "8"))) as p:
        for L in p.imap_unordered(job, J.LAYERS):
            print(L, end=" ", flush=True)
    print("done")
