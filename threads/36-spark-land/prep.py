"""T36 prep (CPU, PRIVATE): per-token sm120tf decode rows in the exact kept-block order of
32-gbdt-sal/private/sm120/blk/sm120tf (same construction as sm120.py prep_tf), checked against the stored bcnt/bsal,
plus the out-of-sample jF fold scores merged into one array per layer.
  -> /tmp/nestquant/36-spark-land/private/tok/L{L}.npz  ids [N,8] u8, sal [N,8] f32 (w^2 |x|^2)
  -> /tmp/nestquant/36-spark-land/private/sc/L{L}.npy   [nb,256] f32, block k = jF score after block k (OOS fold)
  python prep.py"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, "/home/coder/git/nestquant/threads/32-gbdt-sal")
import t32lib as T  # noqa: E402

G, NE, SEQ = 16, 256, 2048
P32 = "/tmp/nestquant/32-gbdt-sal"
SRC = f"{P32}/sm120_traces/serving/predictor/traces"
TFT = f"{P32}/private/trace_sm120"
BLK = f"{P32}/private/sm120/blk/sm120tf"
SC = "/tmp/nestquant/33-search/joint/scores"
OUT = "/tmp/nestquant/36-spark-land/private"
FOLD = {"embedding-drift-monitor": 1, "fin-saccr-rwa": 1, "formal-crypto": 2, "sound-change-cascade": 2,
        "freight-dispatch-shift": 3, "pretrain-shard-corruption": 3}


def rows_index():
    mp = np.load(f"{P32}/private/corpora/sm120tf.map.npz")
    names = [str(x) for x in mp["names"]]
    dms, lens = [], []
    for ti, n in enumerate(names):
        z = np.load(f"{SRC}/{n}.npz")
        wr = mp["row0"][mp["task"] == ti]
        r = (wr[:, None] + np.arange(SEQ)[None]).ravel()
        dm = z["dec"][r]
        dms.append(dm); lens.append(int(dm.sum()))
    dmask = np.concatenate(dms)
    starts = np.r_[0, np.cumsum(lens)]
    keep = np.zeros(int(dmask.sum()), bool)
    for a, b in zip(starts[:-1], starts[1:]):
        keep[a:a + (b - a) // G * G] = True
    return names, dmask, keep


def job(L):
    ids_all, w_all, xn_all = T.load_layer(L, "sm120tf", trace=TFT)
    ids = ids_all[DM][KEEP]; w = w_all[DM][KEEP]; xn = xn_all[DM][KEEP]
    sal = (w.astype(np.float64) ** 2 * xn.astype(np.float64)[:, None]).astype(np.float32)
    nb = ids.shape[0] // G
    b = np.repeat(np.arange(nb), G * 8); e = ids.astype(np.int64).ravel()
    bc = np.bincount(b * NE + e, minlength=nb * NE).reshape(nb, NE)
    bs = np.bincount(b * NE + e, weights=sal.astype(np.float64).ravel(), minlength=nb * NE).reshape(nb, NE)
    d = np.load(f"{BLK}/L{L}.npz")
    okc = bool((bc == d["bcnt"]).all())
    rel = float(np.abs(bs - d["bsal"]).sum() / d["bsal"].sum())
    os.makedirs(f"{OUT}/tok", exist_ok=True); os.makedirs(f"{OUT}/sc", exist_ok=True)
    np.savez(f"{OUT}/tok/L{L}.npz", ids=ids.astype(np.uint8), sal=sal)
    meta = json.load(open(f"{BLK}/meta.json"))
    S = np.zeros((nb, NE), np.float32)
    for n, a, z in zip(meta["chains"], meta["bstart"][:-1], meta["bstart"][1:]):
        if z > a:
            S[a:z] = np.load(f"{SC}/jF{FOLD[n]}_sm120tf/L{L}.npy", mmap_mode="r")[a:z]
    np.save(f"{OUT}/sc/L{L}.npy", S)
    return L, okc, rel


if __name__ == "__main__":
    names, DM, KEEP = rows_index()
    print("tasks", names, "decode rows", int(DM.sum()), "kept", int(KEEP.sum()), flush=True)
    with Pool(int(os.environ.get("NPROC", "12"))) as p:
        for L, okc, rel in p.imap_unordered(job, range(3, 78)):
            print(f"L{L} bcnt_exact {okc} bsal_rel_err {rel:.2e}", flush=True)
