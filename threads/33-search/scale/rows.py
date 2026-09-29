"""training rows (v2 recipe: EMA256 rank 20..120 candidates, next-64 salience / mL) with chain id, + stuck-flag
mem_cur_state as column 9.  rows.py STREAM SUB  -> private/rows/STREAM/{X,y,chain,layer}.npy"""
import os, sys, json
os.environ.setdefault("OMP_NUM_THREADS", "1")
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scalelib as S
stream, SUB = sys.argv[1], int(sys.argv[2])
mL = {int(k): v for k, v in json.load(open("/tmp/nestquant/32-gbdt-sal/models/v2_sal_tweedie1.5.txt.meta.json"))["sal_norm_mL"].items()}
out = f"{S.OUT}/private/rows/{stream}"
os.makedirs(out, exist_ok=True)

def job(L):
    D = S.load(stream, L)
    F, e256 = S.feats(D)
    Fs = S.mem_state(D["bc"], D["bca"], D["nans"], D["segl"], D["sg"], stuck=True)
    F = np.concatenate([F, Fs[..., None]], -1)
    ch = np.zeros(D["bc"].shape[0], np.int16)
    for i, (s, e) in enumerate(D["sg"]):
        ch[s:e] = i
    # SUB: every SUB-th block, offset by layer (like sm120.py rows)
    X, y = S.train_rows(D, F, e256, L, mL=mL[L], sub=SUB)
    fx = np.zeros(S.NE, bool); fx[S.FIXED[L]] = True
    _, v = S.fut(D["bs"], 4, D["sg"])
    if SUB > 1:
        v &= (np.arange(len(v)) % SUB) == (L % SUB)
    c = np.repeat(ch[v], 101)
    np.savez(f"{out}/L{L}.npz", X=X.astype(np.float32), y=y, chain=c)
    return L, len(y)

if __name__ == "__main__":
    with Pool(int(os.environ.get("NPROC", "12"))) as p:
        for L, n in p.imap_unordered(job, S.LAYERS):
            print(L, n, flush=True)
