"""T33b xlatent: shared helpers.  Per-corpus stacked arrays [nb, 75, 256] under D/<corpus>/ (PRIVATE, /tmp only)."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/home/coder/git/nestquant/threads/32-gbdt-sal")
import t32lib as T  # noqa: E402

D = "/tmp/nestquant/33-search/xlatent/data"
LAYERS = T.LAYERS
NL, NE, NF = len(LAYERS), 256, 51
V2 = "/home/coder/git/nestquant/streaming/gbdt_v2sal_p64.txt"
BLK = f"{T.OUT}/private/sm120/blk"


def masks():
    fixed, fdef = T.serve_sets()
    fx = np.zeros((NL, NE), bool); fd = np.zeros((NL, NE), bool)
    for i, L in enumerate(LAYERS):
        fx[i, fixed[L]] = True
        fd[i, [e for e in fdef[L] if e not in set(fixed[L])][:NF]] = True
    return fx, fd


def chains(corpus):
    """list of (start, end) block ranges; calib/heldout: 512-block chains (last may be partial)."""
    if corpus.startswith("sm120"):
        bs = json.load(open(f"{BLK}/{corpus}/meta.json"))["bstart"]
        return [(a, b) for a, b in zip(bs[:-1], bs[1:]) if b > a]
    nb = np.load(f"{D}/{corpus}/bsal.npy", mmap_mode="r").shape[0]
    nbc = T.CHAIN * T.SEQ // T.G
    return [(a, min(a + nbc, nb)) for a in range(0, nb, nbc)]


def load(corpus, name, mmap=True):
    return np.load(f"{D}/{corpus}/{name}.npy", mmap_mode="r" if mmap else None)


def replay(S, fx, fd, sg, hm=0.5):
    """vectorised over layers: S [nb, NL, NE] (score at end of block k, sync) -> serve [nb, NL, NE] bool floating.
    = t32lib.sim_layer(lag=0) / sm120.replay per layer."""
    nb = S.shape[0]
    serve = np.zeros((nb, NL, NE), bool)
    ar = np.arange(NL)[:, None]
    f = np.float32(1 + hm)
    for s, e in sg:
        want = fd.copy()
        for k in range(s, e):
            serve[k] = want
            Sk = np.asarray(S[k], np.float32)
            v = np.where(fx, -np.inf, Sk).astype(np.float32)
            r = want & ~fx
            v = np.where(r, v * f, v)
            tot = np.where(fx, 0, np.maximum(Sk, 0)).sum(1)
            o = np.argsort(-v, 1, kind="stable")[:, :NF]
            nw = np.zeros((NL, NE), bool); nw[ar, o] = True
            nw = np.where((tot <= 0)[:, None], r, nw)
            want = nw & ~fx
    return serve


def metrics(serve, bsal, fx, sg, tf=False):
    """mean over layers of all-slot sal-hot; churn per refresh (calib-style: all consecutive pairs; tf-style: within
    chains).  bsal [nb, NL, NE]."""
    num = np.zeros(NL); den = np.zeros(NL); ch = np.zeros(NL); n = 0
    for s, e in sg:
        sv = serve[s:e]; b = np.asarray(bsal[s:e], np.float64)
        hot = sv | fx[None]
        num += (b * hot).sum((0, 2)); den += b.sum((0, 2))
        if tf:
            ch += (sv[1:] & ~sv[:-1]).sum((0, 2)); n += e - s - 1
    if not tf:
        ch = (serve[1:] & ~serve[:-1]).sum((0, 2)); n = serve.shape[0] - 1
    per = num / den
    return dict(sal=float(per.mean()), churn=float((ch / n).mean()), per_layer=per.tolist(),
                L3_6=float(per[:4].mean()), L7_40=float(per[4:38].mean()), L41_77=float(per[38:].mean()))


def evaluate(S, corpus, hm=0.5, fx=None, fd=None):
    if fx is None:
        fx, fd = masks()
    sg = chains(corpus)
    sv = replay(S, fx, fd, sg, hm)
    return metrics(sv, load(corpus, "bsal"), fx, sg, tf=corpus.startswith("sm120"))
