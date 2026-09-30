#!/usr/bin/env python3
"""prep.py CORPUS: stack bsal [nb,75,256] f32, bcnt u8, v2 score S_v2 f32 (sync, band all), v2 features F9 f16
(calib/heldout only; ema32 ema128 mem_cur_state tok_since_hit hits16 sema32 sema128 sal16 mps128) under D/CORPUS."""
import os
import sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
from multiprocessing import Pool  # noqa: E402
import numpy as np  # noqa: E402
import xlib as X  # noqa: E402
T = X.T
corpus = sys.argv[1]
out = f"{X.D}/{corpus}"
os.makedirs(out, exist_ok=True)


def job(i):
    import lightgbm as lgb
    L = X.LAYERS[i]
    b = lgb.Booster(model_file=X.V2)
    names = b.feature_name()
    if corpus.startswith("sm120"):
        sys.path.insert(0, "/home/coder/git/nestquant/threads/32-gbdt-sal")
        import sm120
        d, meta = sm120.load_blk(corpus, L)
        F = sm120.feats(d, meta, L, set(names))
        bs, bc = d["bsal"].astype(np.float32), d["bcnt"]
        nb = bc.shape[0]
        fx = np.zeros(256, bool); fx[sm120.fixed[L]] = True
        nfx = np.arange(256) if X.K0 else np.flatnonzero(~fx)
        S = np.zeros((nb, 256), np.float32)
        for c0 in range(0, nb, 20000):
            Xm = np.stack([F[f][c0:c0 + 20000][:, nfx] for f in names], -1).reshape(-1, len(names))
            S[c0:c0 + 20000, nfx] = b.predict(Xm, num_threads=2).reshape(-1, len(nfx))
        F9 = None
    else:
        d = np.load(f"{T.OUT}/rows_bandall/{corpus}/L{L}.npz")
        Xm = T.feature_matrix(names, corpus, L, band="all", d=d)
        S = T.score_blocks(b.predict(Xm, num_threads=1), d["cand"], d["top"], d["e256"])
        nb = S.shape[0]
        F9 = np.zeros((nb, 256, 9), np.float16)
        np.put_along_axis(F9, d["cand"].astype(np.int64)[..., None], Xm.reshape(nb, -1, 9).astype(np.float16), 1)
        bs, bc = d["bsal"].astype(np.float32), d["bcnt"]
    return i, S, bs, bc, F9


if __name__ == "__main__":
    tf = corpus.startswith("sm120")
    with Pool(int(os.environ.get("NPROC", "10"))) as p:
        mm = None
        for i, S, bs, bc, F9 in p.imap_unordered(job, range(X.NL)):
            if mm is None:
                nb = S.shape[0]
                mm = {k: np.lib.format.open_memmap(f"{out}/{k}.npy", "w+", dt, (nb, X.NL, 256))
                      for k, dt in (("S_v2", np.float32), ("bsal", np.float32), ("bcnt", np.uint8))}
                if not tf:
                    mm["F9"] = np.lib.format.open_memmap(f"{out}/F9.npy", "w+", np.float16, (nb, X.NL, 256, 9))
            mm["S_v2"][:, i] = S; mm["bsal"][:, i] = bs; mm["bcnt"][:, i] = bc
            if not tf:
                mm["F9"][:, i] = F9
            print(i, flush=True, end=" ")
        for v in mm.values():
            v.flush()
    print("done")
