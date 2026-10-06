"""GLM-5.3 (non-Flash) predictor accuracy with the T37 harness definitions, on the T36 private sm120tf decode traces
(per-token ids/sal + OOS jF scores per block).  Pooled salience coverage over sampled layers, k0 (no fixed set):
  orc  top-H of the same block's salience        lag  top-H of the previous block
  ema  block-level salience EMA (hl 64 tokens), hysteresis 0.5      jF  top-H of jF score after block k-1, hm 0.7"""
import sys
import numpy as np
P = "/tmp/nestquant/36-spark-land/private"
NE, G = 256, 16


def topk_mask(sc, H, cur=None, hm=0.0):
    v = sc * (1 + hm * cur) if cur is not None else sc
    m = np.zeros(NE, bool)
    m[np.lexsort((~cur if cur is not None else np.zeros(NE, bool), -v))[:H]] = True
    return m


acc = {}
for L in map(int, sys.argv[1].split(",")):
    z = np.load(f"{P}/tok/L{L}.npz"); ids = z["ids"].astype(np.int64); sal = z["sal"].astype(np.float64)
    sc = np.load(f"{P}/sc/L{L}.npy")
    nb = len(ids) // G
    B = np.bincount(((np.arange(nb * G) // G)[:, None] * NE + ids[:nb * G]).ravel(), sal[:nb * G].ravel(),
                    minlength=nb * NE).reshape(nb, NE)
    C = np.bincount(((np.arange(nb * G) // G)[:, None] * NE + ids[:nb * G]).ravel(), minlength=nb * NE).reshape(nb, NE)
    tot = B[1:].sum(); ctot = C[1:].sum()
    for H in (45, 77):
        io = np.argsort(-B[1:], 1)[:, :H]                                      # same-block oracle
        o = (np.take_along_axis(B[1:], io, 1).sum(), np.take_along_axis(C[1:], io, 1).sum())
        idx = np.argsort(-B[:-1], 1)[:, :H]
        lag = (np.take_along_axis(B[1:], idx, 1).sum(), np.take_along_axis(C[1:], idx, 1).sum())
        cur = np.zeros(NE, bool); cur[:H] = True; ce = cur.copy(); st = np.zeros(NE); a = 0.5 ** (G / 64)
        jf = np.zeros(2); em = np.zeros(2)
        for k in range(1, nb):
            cur = topk_mask(sc[k - 1].astype(np.float64), H, cur, 0.7); jf += (B[k][cur].sum(), C[k][cur].sum())
            st = st * a + B[k - 1]; ce = topk_mask(st, H, ce, 0.5); em += (B[k][ce].sum(), C[k][ce].sum())
        for n_, v in (("orc", o), ("lag", lag), ("ema", em), ("jF", jf)):
            acc.setdefault((H, n_), np.zeros(4)); acc[(H, n_)] += (v[0], tot, v[1], ctot)
    print(f"L{L}", file=sys.stderr, flush=True)
for (H, n_), v in sorted(acc.items()):
    print(f"H{H} {n_:4s} sal {v[0] / v[1]:.4f} calls {v[2] / v[3]:.4f}")
