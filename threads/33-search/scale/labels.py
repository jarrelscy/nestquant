"""per-block regime labels -> private/labels_STREAM.npz: chain, pos (block idx in chain), rpos (decode tokens since
request start at block's first row; = pos*16 for calib/heldout chains), ans (answer-token fraction per block from the
correct flag), tcls (0 prose, 1 code, 2 math: token-class heuristic over the block's 64-token centred window)."""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scalelib as S
TC = np.load("/tmp/nestquant/32-gbdt-sal/stats/tokclass.npy")
SRC120 = f"{S.SRC}/sm120_traces/serving/predictor/traces"

def tclass(tok_blocks):
    """tok_blocks [nb,16] -> class per block over window of blocks b-1..b+2 (64 tokens)."""
    c = TC[np.clip(tok_blocks, 0, len(TC) - 1)]
    nb = c.shape[0]
    fr = np.stack([(c == k).mean(1) for k in range(5)], 1)       # per block fractions
    k = np.ones(4) / 4
    frw = np.stack([np.convolve(fr[:, j], k, mode="same") for j in range(5)], 1)
    cls = np.zeros(nb, np.int8)
    code = (frw[:, 2] >= 0.12) | (frw[:, 4] >= 0.15)
    math = frw[:, 1] >= 0.15
    cls[code] = 1; cls[math] = 2
    return cls, frw

for stream in sys.argv[1:]:
    D = S.load(stream, 10)
    nb = D["bc"].shape[0]
    ans = D["nans"] / 16.0
    chain = np.zeros(nb, np.int16); pos = np.zeros(nb, np.int32)
    for i, (s, e) in enumerate(D["sg"]):
        chain[s:e] = i; pos[s:e] = np.arange(e - s)
    if stream in ("calib-fit", "glm52-heldout"):
        tok = np.load(f"{S.T.CORP}/{stream}.npy")[: nb * 16].astype(np.int64)
        rpos = pos * 16
        preb = np.zeros(nb, np.int32)
    else:
        mp = np.load(f"{S.SRC}/private/corpora/sm120tf.map.npz")
        names = [str(x) for x in mp["names"]]
        toks, rps, pres = [], [], []
        for ti, n in enumerate(names):
            z = np.load(f"{SRC120}/{n}.npz")
            wr = mp["row0"][mp["task"] == ti]
            r = (wr[:, None] + np.arange(2048)[None]).ravel()
            dm = z["dec"][r]; rd = r[dm]
            if len(rd) < 16:
                continue
            req = z["req"][rd]
            rst = np.r_[True, req[1:] != req[:-1]]
            idx = np.arange(len(rd)); last = np.maximum.accumulate(np.where(rst, idx, 0))
            rp = idx - last
            # prefill rows immediately preceding each decode row (captured rows between consecutive decode rows)
            gap = np.r_[0, np.diff(np.flatnonzero(dm))] - 1
            gap[0] = 0
            nbk = len(rd) // 16
            toks.append(z["tok"][rd][: nbk * 16]); rps.append(rp[: nbk * 16:16]); pres.append(gap[: nbk * 16].reshape(nbk, 16).sum(1))
        tok = np.concatenate(toks).astype(np.int64); rpos = np.concatenate(rps); preb = np.concatenate(pres)
        assert len(rpos) == nb, (len(rpos), nb)
    cls, frw = tclass(tok.reshape(nb, 16))
    np.savez(f"{S.OUT}/private/labels_{stream}.npz", chain=chain, pos=pos, rpos=rpos, ans=ans, tcls=cls, frw=frw, preb=preb)
    print(stream, nb, "ans>0.5", (ans > 0.5).mean(), "tcls", np.bincount(cls, minlength=3) / nb,
          "rpos<1024", (rpos < 1024).mean(), "blocks w/ prefill before", (preb > 0).mean(), flush=True)
