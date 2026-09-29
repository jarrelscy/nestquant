"""T33 draft: shared helpers.  Eval = T32 hot_eval.py semantics (all-slot sal-hot, sync lag 0, hm, band all, chains
4x2048, floating_default at chain start), reusing t32lib.  Extra feature arrays: PRIVATE npz under
/tmp/nestquant/33-search/draft/private/rows_<name>/<corpus>/L{L}.npz with F [nb, 256, k] (band-all cand order) and
a names list."""
import os
import sys

sys.path.insert(0, "/home/coder/git/nestquant/threads/32-gbdt-sal")
import numpy as np  # noqa: E402
import t32lib as T  # noqa: E402

PRIV = "/tmp/nestquant/33-search/draft/private"
V2 = "/tmp/nestquant/32-gbdt-sal/models/v2_sal_tweedie1.5.txt"
BASE = list(T.FEATS5) + list(T.FEATS_V2)
NBC = T.CHAIN * T.SEQ // T.G
FIXED, FDEF = T.serve_sets()
_extra = {}          # feature name -> (rows dir name, index)


def register(rowname, names):
    for i, n in enumerate(names):
        _extra[n] = (rowname, i)


def load_rows(corpus, L):
    return np.load(f"{T.OUT}/rows_bandall/{corpus}/L{L}.npz")


def feats(names, corpus, L, d=None):
    """[nb, 256, k] float32"""
    d = d if d is not None else load_rows(corpus, L)
    base = [n for n in names if n not in _extra]
    cols = {}
    if base:
        M = T.feature_matrix(base, corpus, L, band="all", d=d).reshape(d["X"].shape[0], 256, len(base))
        for i, n in enumerate(base):
            cols[n] = M[..., i]
    grp = {}
    for n in names:
        if n in _extra:
            grp.setdefault(_extra[n][0], []).append(n)
    for rn, ns in grp.items():
        F = np.load(f"{PRIV}/rows_{rn}/{corpus}/L{L}.npz")["F"]
        for n in ns:
            cols[n] = F[..., _extra[n][1]]
    return np.stack([cols[n] for n in names], -1).astype(np.float32)


def evaluate(S, L, bc, bs, hm=0.5):
    sv = T.sim_layer(S, FIXED[L], FDEF[L], nf=51, hm=hm, lag=0)
    ch = float((sv[1:] & ~sv[:-1]).sum(1).mean())
    sv[:, FIXED[L]] = True
    return dict(sal=float((bs * sv).sum() / bs.sum()), cnt=float((bc * sv).sum() / bc.sum()), churn=ch)


def chain_split(nb, val_every=4, val=False):
    """calib-fit split by chain: chain index % val_every == val_every-1 -> validation."""
    ci = np.arange(nb) // NBC
    m = (ci % val_every) == (val_every - 1)
    return m if val else ~m
