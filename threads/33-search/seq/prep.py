"""compact per-corpus arrays: cnt u8 [75,nb,256], saln f16 (sal / m_L, hit units), v2 f32 (v2 score)."""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
from multiprocessing import Pool
import numpy as np
import common as C
corpus = sys.argv[1]
def job(L):
    bc, bs = C.load_blocks(corpus, L)
    return bc.astype(np.uint8), (bs / C.ML[L]).astype(np.float16), (C.v2_scores(corpus, L).astype(np.float32) if os.environ.get("NOV2") != "1" else None)
if __name__ == "__main__":
    with Pool(8) as p:
        r = p.map(job, C.LAYERS)
    os.makedirs(f"{C.WD}/data", exist_ok=True)
    np.save(f"{C.WD}/data/{corpus}.cnt.npy", np.stack([x[0] for x in r]))
    np.save(f"{C.WD}/data/{corpus}.saln.npy", np.stack([x[1] for x in r]))
    if r[0][2] is not None: np.save(f"{C.WD}/data/{corpus}.v2.npy", np.stack([x[2] for x in r]))
    print(corpus, r[0][0].shape)
