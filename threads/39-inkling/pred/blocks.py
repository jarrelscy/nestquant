"""Inkling predictor step 1: per-layer 16-token block matrices from the capture routing traces (PRIVATE, never upload).

  python blocks.py L0 L1 ...   -> /tmp/nq_inkling/pred/blk/L{L}.npz
     cnt [nb,384] u8   routed-slot hits in the block
     sal [nb,384] f32  sum over routed slots of w^2 * xn (w = final gate weight, xn = |x|^2 of the MoE input)
     chain [nb] i32    chain id (one chain = one document segment; features reset at chain start, like a new request)
     pos [nb] i16      block index inside its chain
     split [nb] u8     0 = predictor train, 1 = predictor val (fit windows, 1 in 8 by window), 2 = corpus val (KLD set)
A chain's last partial block (<16 tokens) is kept (it is what serving sees)."""
import os, sys, json
import numpy as np
sys.argv, ARGS = sys.argv[:1], sys.argv[1:]
sys.path.insert(0, "/tmp/nq_inkling/code")
import cap_ink as C

TR = "/tmp/nq_inkling/cap-txt/private/trace"
OUT = "/tmp/nq_inkling/pred/blk"
G, NE, W = 16, 384, 8
os.makedirs(OUT, exist_ok=True)
ROOTS = {"traces": f"{C.CORP}/c2048_traces", "c2048": f"{C.CORP}/c2048"}


def rank_data(r):
    """Data over exactly the windows rank r traced (windows.r{r}of8.json), in trace row order."""
    wl = json.load(open(f"{TR}/windows.r{r}of{W}.json"))["wins"]
    wins = [(g, ROOTS[g], w, s) for g, w, s in wl]
    return C.Data(wins, 0, 1)


DATA = [rank_data(r) for r in range(W)]


def layer(L):
    f = f"{OUT}/L{L}.npz"
    if os.path.exists(f):
        return
    cnt, sal, chain, pos, split = [], [], [], [], []
    cid = 0
    for r in range(W):
        z = np.load(f"{TR}/L{L}.r{r}of{W}.npz")
        ids, w, xn = z["ids"].astype(np.int64), z["w"].astype(np.float32), z["xn"].astype(np.float32)
        d = DATA[r]
        assert ids.shape[0] == d.N, (ids.shape, d.N)
        # window index per row -> predictor split
        wrow = np.zeros(d.N, np.int64)
        for k, (o, root, wi) in enumerate(d.woff):
            wrow[o:o + 2048] = wi * 100 + k
        for sg in d.segs:
            ix = sg["ix"]; n = len(ix); nb = (n + G - 1) // G
            b = np.repeat(np.arange(nb), G)[:n]
            c = np.zeros((nb, NE), np.int32); s = np.zeros((nb, NE), np.float64)
            for j in range(ids.shape[1]):
                np.add.at(c, (b, ids[ix, j]), 1)
                np.add.at(s, (b, ids[ix, j]), (w[ix, j] ** 2 * xn[ix]).astype(np.float64))
            if sg["split"] == "val":
                sp = 2
            else:
                sp = 1 if (wrow[ix[0]] // 100) % 8 == 3 else 0
            cnt.append(np.minimum(c, 255).astype(np.uint8)); sal.append(s.astype(np.float32))
            chain.append(np.full(nb, cid, np.int32)); pos.append(np.arange(nb, dtype=np.int16)); split.append(np.full(nb, sp, np.uint8))
            cid += 1
    np.savez(f + ".part.npz", cnt=np.concatenate(cnt), sal=np.concatenate(sal), chain=np.concatenate(chain),
             pos=np.concatenate(pos), split=np.concatenate(split))
    os.replace(f + ".part.npz", f)
    print(L, "blocks", sum(len(x) for x in chain), "chains", cid, flush=True)


for L in map(int, ARGS):
    layer(L)
