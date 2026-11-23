"""Inkling-2-Mini predictor step 1: per-layer 16-token block tables from the cap_mini routing traces (PRIVATE).

One L4 copy of an expert serves every pass of its layer, so loop layers (6-17, 3 passes) sum hits and salience over
their passes. Output /tmp/nq_mini/pred/blk/L{L}.npz, same schema as the Small builder:
  cnt [nb,128] u8, sal [nb,128] f32 (sum of w^2 * |x|^2), chain [nb] i32, pos [nb] i16,
  split [nb] u8 (0 train, 1 predictor val = window idx % 8 == 3, 2 corpus val).
    python blocks_mini.py 0 1 ... 23"""
import os, sys, json, glob
import numpy as np
sys.argv, ARGS = sys.argv[:1], sys.argv[1:]
sys.path.insert(0, "/tmp/nq_mini/code")
import cap_ink as C

TR = "/tmp/nq_mini/cap-txt/private/trace"
OUT = "/tmp/nq_mini/pred/blk"
G, NE, W = 16, 128, 4
os.makedirs(OUT, exist_ok=True)
DATA = [C.Data(C.windows(0, [761, 1200]), r, W) for r in range(W)]


def layer(L):
    f = f"{OUT}/L{L}.npz"
    if os.path.exists(f):
        return
    cnt, sal, chain, pos, split = [], [], [], [], []
    cid = 0
    for r in range(W):
        d = DATA[r]
        hits = np.zeros((d.N, 8), np.int64) - 1
        parts = sorted(glob.glob(f"{TR}/L{L}.p*.r{r}of{W}.npz"))
        assert parts, (L, r)
        per_pass = []
        for p in parts:
            z = np.load(p)
            per_pass.append((z["rows"].astype(np.int64), z["ids"].astype(np.int64), z["w"].astype(np.float32),
                             z["xn"].astype(np.float32)))
        wrow = np.zeros(d.N, np.int64)
        for k, (o, root, wi) in enumerate(d.woff):
            wrow[o:o + 2048] = wi
        gblk = np.full(d.N, -1, np.int64); nb_tot = 0
        for sg in d.segs:
            ix = sg["ix"]; n = len(ix); nb = (n + G - 1) // G
            gblk[ix] = nb_tot + np.arange(n) // G
            sp = 2 if sg["split"] == "val" else (1 if wrow[ix[0]] % 8 == 3 else 0)
            chain.append(np.full(nb, cid, np.int32)); pos.append(np.arange(nb, dtype=np.int16)); split.append(np.full(nb, sp, np.uint8))
            cid += 1; nb_tot += nb
        c = np.zeros((nb_tot, NE), np.int32); s = np.zeros((nb_tot, NE), np.float64)
        for rows, ids, w, xn in per_pass:
            b = gblk[rows]; m = b >= 0
            for j in range(ids.shape[1]):
                np.add.at(c, (b[m], ids[m, j]), 1)
                np.add.at(s, (b[m], ids[m, j]), (w[m, j] ** 2 * xn[m]).astype(np.float64))
        cnt.append(np.minimum(c, 255).astype(np.uint8)); sal.append(s.astype(np.float32))
    np.savez(f + ".part.npz", cnt=np.concatenate(cnt), sal=np.concatenate(sal), chain=np.concatenate(chain),
             pos=np.concatenate(pos), split=np.concatenate(split))
    os.replace(f + ".part.npz", f)
    print(L, "blocks", sum(len(x) for x in chain), "chains", cid, flush=True)


for L in map(int, ARGS):
    layer(L)
